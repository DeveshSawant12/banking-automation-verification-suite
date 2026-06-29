"""
Fraud Risk Engine — Module 7.

Pure rules-based scoring (no ML model — per locked architecture, this
module combines the VERDICTS already produced by Modules 2, 3, 4, 5, 6
into a single fraud risk score). It does not re-run any detection itself;
it reads persisted results from the database.

LOCKED SCORING RULES (original spec):
    Tampered Aadhaar = +50      Tampered PAN = +50
    Face Mismatch     = +40      Liveness Fail = +40
    Name Mismatch     = +20      DOB Mismatch  = +20

    0-30   Low Risk
    31-60  Medium Risk
    61-100 High Risk

EXTENSIONS BEYOND THE ORIGINAL SPEC (all explicitly locked via dialogue,
not invented silently):

1. INCONCLUSIVE verdicts (possible from Modules 2, 3, 6 when no trained
   model exists or no usable signal could be extracted) are NOT treated
   as "passed". Per locked decision:
     a. HARD OVERRIDE: any INCONCLUSIVE verdict forces
        requires_manual_review=True regardless of numeric score, because
        "we could not verify this" is categorically different from "we
        verified this and it's low-risk" — collapsing the two would let
        an unverifiable case slip through as falsely low-risk.
     b. SOFT POINTS: each INCONCLUSIVE verdict ALSO adds
        INCONCLUSIVE_PENALTY_POINTS (= 20, configurable below) to the
        numeric score, reflecting that an unverifiable signal is itself
        a minor risk indicator, not zero risk.

2. Face Mismatch with two possible comparisons (Aadhaar-vs-selfie AND
   PAN-vs-selfie, when both ID documents are uploaded — per Module 4's
   locked "compare against both" decision): per locked decision, this
   uses WORST-CASE logic. If EITHER comparison is a mismatch (is_match
   = False), the full +40 is applied ONCE (not per-comparison, not
   summed) — a single genuine mismatch is disqualifying regardless of
   how many comparisons were run.

3. Name Mismatch threshold: CrossDocumentResult.name_match_score is a
   continuous RapidFuzz token_sort_ratio (0-100), not already a boolean.
   Per locked decision, name_match_score < NAME_MISMATCH_THRESHOLD (75)
   counts as a mismatch (+20).

SCORE CLAMPING: because INCONCLUSIVE soft points can stack with other
penalties (e.g. an inconclusive Aadhaar tampering check AND a genuine
face mismatch), the raw accumulated point total can exceed 100. The
persisted total_score is clamped to [0, 100] so the risk-level banding
(0-30/31-60/61-100) remains meaningful and bounded, while the full
unclamped contribution list is preserved in score_breakdown for
auditability (Module 8/9/11 need to see exactly what fired, not just the
final clamped number).
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.orm import Session

from app.db.models.cross_document_result import CrossDocumentResult
from app.db.models.document import Document, DocumentType
from app.db.models.face_verification_result import FaceVerificationResult
from app.db.models.fraud_risk_score import FraudRiskScore, RiskLevel
from app.db.models.liveness_result import LivenessResult, LivenessVerdict
from app.db.models.tampering_result import TamperingResult, TamperVerdict
from app.schemas.fraud import FraudRiskAnalysisInternal
from app.services.audit_service import AuditEventType, write_audit_log

logger = logging.getLogger(__name__)

# --- Locked scoring weights (original spec) ---
POINTS_TAMPERED_AADHAAR = 50
POINTS_TAMPERED_PAN = 50
POINTS_FACE_MISMATCH = 40
POINTS_LIVENESS_FAIL = 40
POINTS_NAME_MISMATCH = 20
POINTS_DOB_MISMATCH = 20

# --- Locked extensions ---
INCONCLUSIVE_PENALTY_POINTS = 20
NAME_MISMATCH_THRESHOLD = 75.0  # name_match_score below this = mismatch

# --- Locked risk bands ---
LOW_RISK_MAX = 30
MEDIUM_RISK_MAX = 60
# 61-100 = HIGH_RISK


class FraudRiskServiceError(Exception):
    """Raised when fraud risk scoring cannot be completed."""


def _score_tampering(
    db: Session, kyc_case_id: uuid.UUID, document_type: DocumentType
) -> tuple[int, bool, str]:
    """
    Score a single document's tampering result.

    Returns:
        (points, is_inconclusive, label) where label is a human-readable
        breakdown key, e.g. "tampered_aadhaar" or "inconclusive_aadhaar_tampering".

    If no Document of this type was uploaded for the case, or no
    TamperingResult exists yet, this contributes 0 points but is treated
    as inconclusive (we cannot claim "verified clean" for a check that
    never ran) so the hard-override review flag still applies. This
    matters because the orchestrator should normally guarantee both
    documents and their tampering checks exist before fraud scoring runs,
    but the fraud engine must not silently assume that invariant holds.
    """
    document = (
        db.query(Document)
        .filter(
            Document.kyc_case_id == kyc_case_id,
            Document.document_type == document_type,
        )
        .one_or_none()
    )
    label_prefix = document_type.value.lower()

    if document is None or document.tampering_result is None:
        logger.warning(
            "No tampering result found for %s on case %s — treating as "
            "INCONCLUSIVE (cannot score a check that never ran).",
            document_type.value,
            kyc_case_id,
        )
        return INCONCLUSIVE_PENALTY_POINTS, True, f"inconclusive_{label_prefix}_tampering"

    verdict = document.tampering_result.verdict

    if verdict == TamperVerdict.TAMPERED:
        points = POINTS_TAMPERED_AADHAAR if document_type == DocumentType.AADHAAR else POINTS_TAMPERED_PAN
        return points, False, f"tampered_{label_prefix}"

    if verdict == TamperVerdict.INCONCLUSIVE:
        return INCONCLUSIVE_PENALTY_POINTS, True, f"inconclusive_{label_prefix}_tampering"

    # REAL
    return 0, False, f"{label_prefix}_verified_real"


def _score_face_verification(
    db: Session, kyc_case_id: uuid.UUID
) -> tuple[int, dict]:
    """
    Score face verification using worst-case logic across all
    FaceVerificationResult rows for this case (one per available ID
    document: Aadhaar and/or PAN, per Module 4's locked "compare against
    both" decision).

    Per locked decision: if EITHER comparison mismatches, apply the full
    +40 ONCE. If no FaceVerificationResult rows exist at all (face
    verification never ran), this is reported as a breakdown note but
    intentionally contributes 0 points here — unlike tampering/liveness,
    face verification has no INCONCLUSIVE state of its own in the current
    schema (is_match is a plain boolean); a fully missing check is a
    pipeline-ordering problem for the orchestrator to prevent, not
    something this scoring function should invent a penalty for.
    """
    results = (
        db.query(FaceVerificationResult)
        .filter(FaceVerificationResult.kyc_case_id == kyc_case_id)
        .all()
    )

    detail = {
        "comparisons_run": len(results),
        "comparison_results": [
            {"id_document_id": str(r.id_document_id), "is_match": r.is_match, "match_percentage": r.match_percentage}
            for r in results
        ],
    }

    if not results:
        logger.warning(
            "No face verification results found for case %s — "
            "contributing 0 points (orchestrator should ensure this "
            "check ran before fraud scoring).",
            kyc_case_id,
        )
        return 0, detail

    any_mismatch = any(not r.is_match for r in results)
    points = POINTS_FACE_MISMATCH if any_mismatch else 0
    return points, detail


def _score_liveness(db: Session, kyc_case_id: uuid.UUID) -> tuple[int, bool]:
    """
    Score liveness detection.

    Returns:
        (points, is_inconclusive)
    """
    result = (
        db.query(LivenessResult)
        .filter(LivenessResult.kyc_case_id == kyc_case_id)
        .one_or_none()
    )

    if result is None:
        logger.warning(
            "No liveness result found for case %s — treating as "
            "INCONCLUSIVE.",
            kyc_case_id,
        )
        return INCONCLUSIVE_PENALTY_POINTS, True

    if result.verdict == LivenessVerdict.SPOOF:
        return POINTS_LIVENESS_FAIL, False
    if result.verdict == LivenessVerdict.INCONCLUSIVE:
        return INCONCLUSIVE_PENALTY_POINTS, True

    # LIVE
    return 0, False


def _score_cross_document(
    db: Session, kyc_case_id: uuid.UUID
) -> tuple[int, int, dict]:
    """
    Score name and DOB mismatch from CrossDocumentResult.

    Returns:
        (name_points, dob_points, detail)
    """
    result = (
        db.query(CrossDocumentResult)
        .filter(CrossDocumentResult.kyc_case_id == kyc_case_id)
        .one_or_none()
    )

    if result is None:
        logger.warning(
            "No cross-document result found for case %s — contributing "
            "0 points for name/DOB checks.",
            kyc_case_id,
        )
        return 0, 0, {"ran": False}

    name_mismatch = result.name_match_score < NAME_MISMATCH_THRESHOLD
    dob_mismatch = not result.dob_match

    name_points = POINTS_NAME_MISMATCH if name_mismatch else 0
    dob_points = POINTS_DOB_MISMATCH if dob_mismatch else 0

    detail = {
        "ran": True,
        "name_match_score": result.name_match_score,
        "name_mismatch": name_mismatch,
        "dob_match": result.dob_match,
        "dob_mismatch": dob_mismatch,
    }
    return name_points, dob_points, detail


def _determine_risk_level(clamped_score: int) -> RiskLevel:
    if clamped_score <= LOW_RISK_MAX:
        return RiskLevel.LOW
    if clamped_score <= MEDIUM_RISK_MAX:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def compute_fraud_risk_score(
    db: Session, kyc_case_id: uuid.UUID
) -> FraudRiskAnalysisInternal:
    """
    Compute the full fraud risk score for a KYC case by reading all
    persisted verification results (Modules 2, 3, 4, 5, 6) and applying
    the locked scoring rules plus the locked INCONCLUSIVE/face-mismatch
    extensions documented in this module's docstring.

    This function does NOT raise on missing sub-results (e.g. a check
    that hasn't run yet) — each `_score_*` helper handles that by
    treating it as INCONCLUSIVE/zero-contribution and logging a warning,
    since the fraud engine's job is to score what's available, not to
    enforce pipeline ordering (that is the orchestrator's responsibility,
    built once all modules exist).
    """
    breakdown: dict = {}
    raw_total = 0
    any_inconclusive = False

    aadhaar_points, aadhaar_inconclusive, aadhaar_label = _score_tampering(
        db, kyc_case_id, DocumentType.AADHAAR
    )
    breakdown[aadhaar_label] = aadhaar_points
    raw_total += aadhaar_points
    any_inconclusive = any_inconclusive or aadhaar_inconclusive

    pan_points, pan_inconclusive, pan_label = _score_tampering(
        db, kyc_case_id, DocumentType.PAN
    )
    breakdown[pan_label] = pan_points
    raw_total += pan_points
    any_inconclusive = any_inconclusive or pan_inconclusive

    face_points, face_detail = _score_face_verification(db, kyc_case_id)
    if face_points > 0:
        breakdown["face_mismatch"] = face_points
    raw_total += face_points
    breakdown["face_verification_detail"] = face_detail

    liveness_points, liveness_inconclusive = _score_liveness(db, kyc_case_id)
    if liveness_inconclusive:
        breakdown["inconclusive_liveness"] = liveness_points
    elif liveness_points > 0:
        breakdown["liveness_fail"] = liveness_points
    raw_total += liveness_points
    any_inconclusive = any_inconclusive or liveness_inconclusive

    name_points, dob_points, cross_doc_detail = _score_cross_document(db, kyc_case_id)
    if name_points > 0:
        breakdown["name_mismatch"] = name_points
    if dob_points > 0:
        breakdown["dob_mismatch"] = dob_points
    raw_total += name_points + dob_points
    breakdown["cross_document_detail"] = cross_doc_detail

    breakdown["raw_total_before_clamp"] = raw_total

    clamped_score = max(0, min(raw_total, 100))
    risk_level = _determine_risk_level(clamped_score)

    requires_manual_review = any_inconclusive or risk_level == RiskLevel.HIGH

    logger.info(
        "Fraud risk computed for case %s: raw_total=%d clamped=%d "
        "risk_level=%s requires_manual_review=%s",
        kyc_case_id,
        raw_total,
        clamped_score,
        risk_level.value,
        requires_manual_review,
    )

    return FraudRiskAnalysisInternal(
        total_score=clamped_score,
        risk_level=risk_level.value,
        score_breakdown=breakdown,
        requires_manual_review=requires_manual_review,
    )


def persist_fraud_risk_score(
    db: Session, kyc_case_id: uuid.UUID, analysis: FraudRiskAnalysisInternal
) -> FraudRiskScore:
    """
    Persist a fraud risk score. One row per kyc_case_id (unique
    constraint) — re-scoring requires explicit deletion of the prior
    record, consistent with other Module services in this project.
    """
    existing = (
        db.query(FraudRiskScore)
        .filter(FraudRiskScore.kyc_case_id == kyc_case_id)
        .one_or_none()
    )
    if existing is not None:
        raise FraudRiskServiceError(
            f"Fraud risk score already exists for case {kyc_case_id}."
        )

    score = FraudRiskScore(
        kyc_case_id=kyc_case_id,
        total_score=analysis.total_score,
        risk_level=RiskLevel(analysis.risk_level),
        score_breakdown=analysis.score_breakdown,
        requires_manual_review=analysis.requires_manual_review,
    )
    db.add(score)
    db.commit()
    db.refresh(score)

    # Module 9 wiring: fraud risk scoring is the most decision-critical
    # audit entry in the pipeline, so risk_score and decision are
    # populated meaningfully here (decision reflects the scoring outcome,
    # not yet the final human/orchestrator approve/reject call, which
    # gets its own KYC_CASE_DECISION_MADE entry when that exists).
    write_audit_log(
        db,
        AuditEventType.FRAUD_RISK_SCORED,
        kyc_case_id=kyc_case_id,
        verification_status=analysis.risk_level,
        risk_score=analysis.total_score,
        decision="REQUIRES_REVIEW" if analysis.requires_manual_review else "AUTO_PASS",
        metadata_json={
            "score_breakdown": analysis.score_breakdown,
            "requires_manual_review": analysis.requires_manual_review,
        },
    )

    return score


def run_and_persist_fraud_risk_scoring(
    db: Session, kyc_case_id: uuid.UUID
) -> FraudRiskScore:
    """Convenience wrapper: computes and persists the fraud risk score."""
    analysis = compute_fraud_risk_score(db, kyc_case_id)
    return persist_fraud_risk_score(db, kyc_case_id, analysis)
