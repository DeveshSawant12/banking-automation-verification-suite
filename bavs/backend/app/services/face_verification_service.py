"""
Face Verification Service — Module 4 orchestration layer.

Per locked decision: compares the customer selfie against BOTH the
Aadhaar photo and the PAN photo when both documents were uploaded for a
KYC case (not just Aadhaar alone, despite the original spec's pipeline
diagram only mentioning Aadhaar — confirmed explicitly with the project
owner). If only one ID document type was uploaded, verification runs
against whichever is available.

This produces ONE FaceVerificationResult row per ID document compared
(so up to two rows per kyc_case_id: one for Aadhaar-vs-selfie, one for
PAN-vs-selfie). overall_is_match for the case is the AND of all
individual comparisons -- a customer must match every ID photo they
submitted, not just one.

No silent failure paths: if no face can be extracted from either the ID
photo or the selfie, this raises NoFaceDetectedError (uncaught here,
deliberately, mirroring the ModelNotTrainedError pattern from Module
2/3) so the orchestrator can route the case to REVIEW_REQUIRED rather
than silently skipping face verification.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.orm import Session

from app.db.models.document import Document, DocumentType
from app.db.models.face_verification_result import FaceVerificationResult
from app.ml.face.deepface_engine import (
    DETECTOR_BACKEND,
    RECOGNITION_MODEL,
    NoFaceDetectedError,
    extract_face_embedding,
    verify_faces,
)
from app.services.audit_service import AuditEventType, write_audit_log
from app.utils.image_utils import ImageLoadError, load_image_from_bytes

logger = logging.getLogger(__name__)


class FaceVerificationServiceError(Exception):
    """Raised on unrecoverable errors during face verification."""


def compare_id_photo_to_selfie(
    id_photo_bytes: bytes, selfie_bytes: bytes
) -> dict:
    """
    Run face extraction on both images and compute the match result.

    Raises:
        FaceVerificationServiceError: on image loading failure.
        NoFaceDetectedError: if either image has zero detectable faces.
            Propagates uncaught -- the caller must route to REVIEW_REQUIRED.
    """
    try:
        id_photo_image = load_image_from_bytes(id_photo_bytes)
        selfie_image = load_image_from_bytes(selfie_bytes)
    except ImageLoadError as exc:
        raise FaceVerificationServiceError(f"Could not load image: {exc}") from exc

    id_photo_face = extract_face_embedding(id_photo_image)
    selfie_face = extract_face_embedding(selfie_image)

    match_result = verify_faces(id_photo_face.embedding, selfie_face.embedding)

    return {
        "cosine_similarity": match_result.cosine_similarity,
        "match_percentage": match_result.match_percentage,
        "is_match": match_result.is_match,
        "id_photo_face_confidence": id_photo_face.face_confidence,
        "selfie_face_confidence": selfie_face.face_confidence,
    }


def persist_face_verification_result(
    db: Session,
    kyc_case_id: uuid.UUID,
    id_document_id: uuid.UUID,
    selfie_document_id: uuid.UUID,
    cosine_similarity: float,
    match_percentage: float,
    is_match: bool,
) -> FaceVerificationResult:
    """
    Persist a single ID-photo-vs-selfie comparison result. Does NOT
    enforce uniqueness per kyc_case_id (multiple rows per case are
    expected -- one per ID document type compared), but DOES enforce
    that the same id_document_id is not compared twice for the same
    case, which would indicate a caller bug (duplicate pipeline run).
    """
    existing = (
        db.query(FaceVerificationResult)
        .filter(
            FaceVerificationResult.kyc_case_id == kyc_case_id,
            FaceVerificationResult.id_document_id == id_document_id,
        )
        .one_or_none()
    )
    if existing is not None:
        raise FaceVerificationServiceError(
            f"Face verification already exists for case {kyc_case_id} "
            f"against document {id_document_id}."
        )

    result = FaceVerificationResult(
        kyc_case_id=kyc_case_id,
        id_document_id=id_document_id,
        selfie_document_id=selfie_document_id,
        cosine_similarity=cosine_similarity,
        match_percentage=match_percentage,
        is_match=is_match,
        detector_backend=DETECTOR_BACKEND,
        model_name=RECOGNITION_MODEL,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def run_face_verification_for_case(
    db: Session,
    kyc_case_id: uuid.UUID,
    document_bytes_by_id: dict[uuid.UUID, bytes],
) -> list[FaceVerificationResult]:
    """
    Run face verification for an entire KYC case: fetches all AADHAAR,
    PAN, and SELFIE documents belonging to the case, then compares the
    selfie against every available ID photo (Aadhaar and/or PAN).

    Args:
        document_bytes_by_id: mapping of Document.id -> raw file bytes for
            every document needed (caller is responsible for fetching
            these from Cloudflare R2 via storage_service -- this function
            stays storage-agnostic and pure with respect to byte content).

    Returns:
        List of persisted FaceVerificationResult rows (one per ID
        document compared against the selfie).

    Raises:
        FaceVerificationServiceError: if no SELFIE document exists for
            the case, or no ID document (AADHAAR/PAN) exists at all --
            face verification cannot proceed without both.
        NoFaceDetectedError: propagates uncaught from
            compare_id_photo_to_selfie if any image lacks a detectable
            face. Caller must route to REVIEW_REQUIRED.
    """
    documents = db.query(Document).filter(Document.kyc_case_id == kyc_case_id).all()

    selfie_doc = next(
        (d for d in documents if d.document_type == DocumentType.SELFIE), None
    )
    if selfie_doc is None:
        raise FaceVerificationServiceError(
            f"No SELFIE document found for KYC case {kyc_case_id}. "
            f"Face verification requires a selfie."
        )

    id_docs = [
        d
        for d in documents
        if d.document_type in (DocumentType.AADHAAR, DocumentType.PAN)
    ]
    if not id_docs:
        raise FaceVerificationServiceError(
            f"No AADHAAR or PAN document found for KYC case {kyc_case_id}. "
            f"Face verification requires at least one ID document."
        )

    if selfie_doc.id not in document_bytes_by_id:
        raise FaceVerificationServiceError(
            f"Selfie document {selfie_doc.id} bytes not provided to "
            f"run_face_verification_for_case."
        )
    selfie_bytes = document_bytes_by_id[selfie_doc.id]

    results: list[FaceVerificationResult] = []
    for id_doc in id_docs:
        if id_doc.id not in document_bytes_by_id:
            raise FaceVerificationServiceError(
                f"ID document {id_doc.id} ({id_doc.document_type}) bytes "
                f"not provided to run_face_verification_for_case."
            )
        id_photo_bytes = document_bytes_by_id[id_doc.id]

        # NoFaceDetectedError propagates uncaught -- by design.
        comparison = compare_id_photo_to_selfie(id_photo_bytes, selfie_bytes)

        result = persist_face_verification_result(
            db,
            kyc_case_id=kyc_case_id,
            id_document_id=id_doc.id,
            selfie_document_id=selfie_doc.id,
            cosine_similarity=comparison["cosine_similarity"],
            match_percentage=comparison["match_percentage"],
            is_match=comparison["is_match"],
        )
        results.append(result)

        logger.info(
            "Face verification: case=%s id_doc=%s(%s) match=%s percentage=%.2f",
            kyc_case_id,
            id_doc.id,
            id_doc.document_type,
            comparison["is_match"],
            comparison["match_percentage"],
        )

    # Module 9 wiring: write a single case-level audit entry summarizing
    # all comparisons. face_match_pct uses the average match_percentage
    # across all comparisons run (Aadhaar-vs-selfie and/or PAN-vs-selfie)
    # since audit_logs.face_match_pct is a single float column, not a
    # per-comparison breakdown -- the full per-comparison detail is still
    # available in metadata_json for anyone who needs it.
    overall_match = case_overall_face_match(results)
    avg_match_pct = (
        sum(r.match_percentage for r in results) / len(results) if results else 0.0
    )
    write_audit_log(
        db,
        AuditEventType.FACE_VERIFICATION_COMPLETED,
        kyc_case_id=kyc_case_id,
        verification_status="MATCH" if overall_match else "MISMATCH",
        face_match_pct=avg_match_pct,
        metadata_json={
            "comparisons": [
                {
                    "id_document_id": str(r.id_document_id),
                    "is_match": r.is_match,
                    "match_percentage": r.match_percentage,
                }
                for r in results
            ],
            "overall_match": overall_match,
        },
    )

    return results


def case_overall_face_match(results: list[FaceVerificationResult]) -> bool:
    """
    A KYC case passes face verification only if EVERY ID-document
    comparison matched. If the customer uploaded both Aadhaar and PAN,
    their selfie must match both photos -- a mismatch against either is
    treated as a face verification failure for the case.
    """
    if not results:
        return False
    return all(r.is_match for r in results)
