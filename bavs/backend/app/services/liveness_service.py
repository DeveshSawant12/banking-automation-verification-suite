"""
Liveness Detection Service — Module 6 orchestration layer.

Pipeline (per locked spec):
    Customer Video -> Frame Extraction -> Per-Frame MediaPipe Analysis
    -> Sequence Aggregation (blink/head-movement/smile) -> Verdict
    (LIVE or SPOOF)

Composes app/ml/liveness/video_sequence_analyzer.py (frame extraction +
aggregation) and app/ml/liveness/mediapipe_liveness.py (per-frame
detection), neither of which is duplicated here.

VERDICT LOGIC: a person is judged LIVE if at least one of the three
liveness actions (blink, head movement, smile) was detected somewhere in
the sequence. This is a deliberately permissive AND-of-conditions-is-NOT-
required design: requiring ALL THREE actions in one short clip would
produce a high false-rejection rate for genuine users who, say, didn't
smile on cue. Requiring at least ONE clear involuntary/voluntary action
balances spoof-resistance (a static photo or video replay cannot produce
ANY of these signals) against usability. This threshold logic lives here
in the service layer (not buried in video_sequence_analyzer.py) so it can
be tuned independently of the underlying signal-extraction code.

SAFETY BEHAVIOR (consistent with Modules 2/3's ModelNotTrainedError
pattern): if the MediaPipe FaceLandmarker .task model file is not present,
this service does NOT fabricate a LIVE/SPOOF verdict. It persists
INCONCLUSIVE and the caller must route the case to REVIEW_REQUIRED.
Likewise, if the uploaded video contains no usable face in any frame
(NoFaceInFrameError, fully exhausted), this is also INCONCLUSIVE, not
SPOOF — "we could not assess this video" is a different claim than "this
video is a fraud attempt", and the fraud risk engine downstream should not
conflate the two.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.orm import Session

from app.db.models.kyc_case import KycCase
from app.db.models.liveness_result import LivenessResult, LivenessVerdict
from app.ml.liveness.mediapipe_liveness import LivenessModelNotAvailableError, NoFaceInFrameError
from app.ml.liveness.video_sequence_analyzer import (
    LivenessSequenceResult,
    VideoProcessingError,
    analyze_frame_sequence,
    extract_frames_from_video,
)
from app.schemas.liveness import LivenessAnalysisInternal
from app.services.audit_service import AuditEventType, write_audit_log

logger = logging.getLogger(__name__)


class LivenessServiceError(Exception):
    """Raised on unrecoverable errors during liveness detection."""


def _determine_verdict(sequence_result: LivenessSequenceResult) -> str:
    """
    Apply the verdict threshold: LIVE if at least one liveness action
    (blink, head movement, or smile) was detected anywhere in the
    sequence; SPOOF if none were detected despite having usable frames
    with a detected face.
    """
    any_action_detected = (
        sequence_result.blink_detected
        or sequence_result.head_movement_detected
        or sequence_result.smile_detected
    )
    return LivenessVerdict.LIVE.value if any_action_detected else LivenessVerdict.SPOOF.value


def analyze_liveness(video_path: str) -> LivenessAnalysisInternal:
    """
    Run the full liveness detection pipeline on an uploaded video file.

    Args:
        video_path: local filesystem path to the uploaded video (already
            downloaded from R2 by the caller — this function does not
            perform storage I/O itself, consistent with other ml/service
            modules in this project)

    Returns:
        LivenessAnalysisInternal with verdict and all aggregated signals.

    Raises:
        LivenessServiceError: on video processing failure.
        LivenessModelNotAvailableError: if the MediaPipe model file is
            missing. Callers MUST catch this and treat the case as
            REVIEW_REQUIRED, never as an implicit verdict.
    """
    try:
        frames = extract_frames_from_video(video_path)
    except VideoProcessingError as exc:
        raise LivenessServiceError(f"Video processing failed: {exc}") from exc

    # NoFaceInFrameError and LivenessModelNotAvailableError propagate
    # uncaught here, by design -- see module docstring.
    sequence_result = analyze_frame_sequence(frames)

    verdict = _determine_verdict(sequence_result)

    return LivenessAnalysisInternal(
        verdict=verdict,
        blink_detected=sequence_result.blink_detected,
        head_movement_detected=sequence_result.head_movement_detected,
        smile_detected=sequence_result.smile_detected,
        max_yaw_delta_degrees=sequence_result.max_yaw_delta_degrees,
        total_frames_analyzed=sequence_result.total_frames_analyzed,
        frames_with_face=sequence_result.frames_with_face,
        frames_without_face=sequence_result.frames_without_face,
    )


def persist_liveness_result(
    db: Session,
    kyc_case_id: uuid.UUID,
    analysis: LivenessAnalysisInternal,
    raw_metrics_json: dict,
) -> LivenessResult:
    """
    Persist a liveness analysis result. One row per kyc_case_id
    (unique constraint) — re-running liveness for the same case requires
    explicit deletion of the prior record, consistent with
    ocr_service.persist_ocr_extraction's pattern.
    """
    case = db.get(KycCase, kyc_case_id)
    if case is None:
        raise LivenessServiceError(f"KycCase {kyc_case_id} does not exist.")

    existing = (
        db.query(LivenessResult)
        .filter(LivenessResult.kyc_case_id == kyc_case_id)
        .one_or_none()
    )
    if existing is not None:
        raise LivenessServiceError(
            f"Liveness result already exists for case {kyc_case_id}."
        )

    result = LivenessResult(
        kyc_case_id=kyc_case_id,
        verdict=LivenessVerdict(analysis.verdict),
        blink_detected=analysis.blink_detected,
        head_movement_detected=analysis.head_movement_detected,
        smile_detected=analysis.smile_detected,
        max_yaw_delta_degrees=analysis.max_yaw_delta_degrees,
        total_frames_analyzed=analysis.total_frames_analyzed,
        frames_with_face=analysis.frames_with_face,
        frames_without_face=analysis.frames_without_face,
        raw_metrics_json=raw_metrics_json,
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    # Module 9 wiring: this function is the single convergence point for
    # all three caller paths in run_and_persist_liveness_check (success,
    # model-unavailable-INCONCLUSIVE, no-face-INCONCLUSIVE), so writing
    # the audit entry here covers all of them without duplicating the
    # wiring three times.
    write_audit_log(
        db,
        AuditEventType.LIVENESS_CHECK_COMPLETED,
        kyc_case_id=kyc_case_id,
        verification_status=result.verdict.value,
        metadata_json={
            "blink_detected": result.blink_detected,
            "head_movement_detected": result.head_movement_detected,
            "smile_detected": result.smile_detected,
            "total_frames_analyzed": result.total_frames_analyzed,
        },
    )

    return result


def run_and_persist_liveness_check(
    db: Session, kyc_case_id: uuid.UUID, video_path: str
) -> LivenessResult:
    """
    Convenience wrapper: runs the full pipeline and persists the result.

    Both failure modes that must not produce a fabricated LIVE/SPOOF
    verdict are handled here:
      1. LivenessModelNotAvailableError (.task model file missing)
      2. NoFaceInFrameError fully exhausted (video has zero usable frames
         with a detectable face)
    Both result in a persisted INCONCLUSIVE verdict with zero confidence
    signal, and the orchestrator must route the case to REVIEW_REQUIRED.
    """
    try:
        analysis = analyze_liveness(video_path)
        raw_metrics = {
            "blink_detected": analysis.blink_detected,
            "head_movement_detected": analysis.head_movement_detected,
            "smile_detected": analysis.smile_detected,
            "max_yaw_delta_degrees": analysis.max_yaw_delta_degrees,
            "total_frames_analyzed": analysis.total_frames_analyzed,
            "frames_with_face": analysis.frames_with_face,
            "frames_without_face": analysis.frames_without_face,
        }
        return persist_liveness_result(db, kyc_case_id, analysis, raw_metrics)

    except LivenessModelNotAvailableError as exc:
        logger.warning(
            "MediaPipe FaceLandmarker model unavailable for case %s: %s. "
            "Persisting INCONCLUSIVE verdict; case must be routed to "
            "REVIEW_REQUIRED by the orchestrator.",
            kyc_case_id,
            exc,
        )
        return _persist_inconclusive(db, kyc_case_id, reason=str(exc))

    except NoFaceInFrameError as exc:
        logger.warning(
            "No usable face detected in any frame for case %s: %s. "
            "Persisting INCONCLUSIVE verdict; case must be routed to "
            "REVIEW_REQUIRED by the orchestrator.",
            kyc_case_id,
            exc,
        )
        return _persist_inconclusive(db, kyc_case_id, reason=str(exc))


def _persist_inconclusive(
    db: Session, kyc_case_id: uuid.UUID, reason: str
) -> LivenessResult:
    """Shared helper for persisting an INCONCLUSIVE result with a logged reason."""
    analysis = LivenessAnalysisInternal(
        verdict=LivenessVerdict.INCONCLUSIVE.value,
        blink_detected=False,
        head_movement_detected=False,
        smile_detected=False,
        max_yaw_delta_degrees=0.0,
        total_frames_analyzed=0,
        frames_with_face=0,
        frames_without_face=0,
    )
    return persist_liveness_result(
        db, kyc_case_id, analysis, raw_metrics_json={"inconclusive_reason": reason}
    )
