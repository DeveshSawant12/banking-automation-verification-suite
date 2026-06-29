"""
Video frame extraction and temporal signal aggregation for liveness
detection.

Takes an uploaded short video (.mp4/.webm per locked decision), extracts
frames at a fixed sampling rate via OpenCV's VideoCapture, runs
mediapipe_liveness.analyze_frame on each, and aggregates the per-frame
signals into action-level verdicts: was a blink detected ANYWHERE in the
sequence, was a head turn (left/right) detected, was a smile detected.

Head turn detection works by tracking yaw_degrees across consecutive
frames and checking whether the yaw moved past
HEAD_YAW_MOVEMENT_THRESHOLD_DEGREES in a consistent direction relative to
a baseline (the first frame's yaw, assumed roughly frontal) — this is a
deliberate design choice: a single frame's absolute yaw doesn't indicate
a "turn", only a *change* in yaw across the sequence does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.ml.liveness.mediapipe_liveness import (
    HEAD_YAW_MOVEMENT_THRESHOLD_DEGREES,
    FrameAnalysis,
    NoFaceInFrameError,
    analyze_frame,
)

logger = logging.getLogger(__name__)

DEFAULT_FRAME_SAMPLE_RATE = 5  # analyze every Nth frame (full video analysis is expensive)
MAX_FRAMES_TO_ANALYZE = 60  # hard cap to bound processing time per video


class VideoProcessingError(Exception):
    """Raised when the uploaded video cannot be opened or read."""


def extract_frames_from_video(
    video_path: str | Path, sample_rate: int = DEFAULT_FRAME_SAMPLE_RATE
) -> list[np.ndarray]:
    """
    Extract frames from a video file at the given sampling rate (every
    Nth frame), bounded by MAX_FRAMES_TO_ANALYZE.

    Args:
        video_path: path to the uploaded video file on disk
        sample_rate: analyze every Nth frame (default 5 — e.g. for a
            30fps video, this analyzes 6 frames per second)

    Returns:
        List of BGR numpy frame arrays.

    Raises:
        VideoProcessingError: if the video cannot be opened or contains
            zero readable frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"Could not open video file at {video_path}.")

    frames: list[np.ndarray] = []
    frame_index = 0

    try:
        while len(frames) < MAX_FRAMES_TO_ANALYZE:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % sample_rate == 0:
                frames.append(frame)
            frame_index += 1
    finally:
        cap.release()

    if not frames:
        raise VideoProcessingError(
            f"No readable frames extracted from video at {video_path}. "
            f"File may be corrupted, empty, or in an unsupported codec."
        )

    return frames


@dataclass
class LivenessSequenceResult:
    blink_detected: bool
    head_movement_detected: bool
    max_yaw_delta_degrees: float  # magnitude of largest head-turn detected, signed
    smile_detected: bool
    total_frames_analyzed: int
    frames_with_face: int
    frames_without_face: int
    raw_frame_analyses: list[dict] = field(default_factory=list)


def analyze_frame_sequence(frames: list[np.ndarray]) -> LivenessSequenceResult:
    """
    Run per-frame MediaPipe analysis across an entire frame sequence and
    aggregate into sequence-level action verdicts.

    Frames where no face is detected are SKIPPED (not treated as a fatal
    error for the whole sequence) and counted in frames_without_face —
    natural video capture often has a few frames with motion blur or the
    face briefly out of frame, which shouldn't fail the whole liveness
    check. However, if NO frames in the entire sequence have a detectable
    face, this is escalated to NoFaceInFrameError (uncaught, propagates
    to caller) since that indicates a fundamentally unusable video, not
    a momentary blip.
    """
    analyses: list[FrameAnalysis] = []
    frames_without_face = 0

    for idx, frame in enumerate(frames):
        try:
            analysis = analyze_frame(frame, frame_index=idx)
            analyses.append(analysis)
        except NoFaceInFrameError:
            frames_without_face += 1
            logger.debug("No face detected in frame %d, skipping.", idx)
            continue

    if not analyses:
        raise NoFaceInFrameError(
            f"No face detected in ANY of the {len(frames)} analyzed frames. "
            f"Video does not contain a usable face sequence."
        )

    blink_detected = any(a.blink_detected for a in analyses)
    smile_detected = any(a.smile_detected for a in analyses)

    baseline_yaw = analyses[0].yaw_degrees
    max_left_delta = 0.0
    max_right_delta = 0.0
    for a in analyses[1:]:
        delta = a.yaw_degrees - baseline_yaw
        if delta > max_left_delta:
            max_left_delta = delta
        if -delta > max_right_delta:
            max_right_delta = -delta

    head_movement_detected = (
        max_left_delta > HEAD_YAW_MOVEMENT_THRESHOLD_DEGREES
        or max_right_delta > HEAD_YAW_MOVEMENT_THRESHOLD_DEGREES
    )
    # NOTE: we deliberately do NOT label this "left" or "right" here. The
    # sign of yaw (positive vs negative) maps to physical left/right
    # depending on the camera's coordinate convention (a front-facing
    # selfie camera is often mirrored relative to a world-facing camera),
    # which this backend module cannot determine without explicit
    # frontend coordination on how the video was captured/encoded.
    # Reporting a guessed direction here risks silently inverting
    # left/right for some camera setups -- a correctness issue in a
    # fraud-relevant signal. We report the signed magnitude instead;
    # if/when the frontend confirms its camera convention, direction
    # labeling can be added as an explicit, verified mapping rather than
    # an assumption.
    max_yaw_delta = max_left_delta if max_left_delta >= max_right_delta else -max_right_delta

    return LivenessSequenceResult(
        blink_detected=blink_detected,
        head_movement_detected=head_movement_detected,
        max_yaw_delta_degrees=max_yaw_delta,
        smile_detected=smile_detected,
        total_frames_analyzed=len(frames),
        frames_with_face=len(analyses),
        frames_without_face=frames_without_face,
        raw_frame_analyses=[
            {
                "frame_index": a.frame_index,
                "eye_blink_left": a.eye_blink_left,
                "eye_blink_right": a.eye_blink_right,
                "mouth_smile_left": a.mouth_smile_left,
                "mouth_smile_right": a.mouth_smile_right,
                "yaw_degrees": a.yaw_degrees,
            }
            for a in analyses
        ],
    )
