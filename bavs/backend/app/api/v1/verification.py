"""
Verification router — read-only results endpoints.

GET /verification/{case_id}/ocr
GET /verification/{case_id}/tampering
GET /verification/{case_id}/face-match
GET /verification/{case_id}/cross-document
GET /verification/{case_id}/liveness
GET /verification/{case_id}/gradcam
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models.document import Document, DocumentType
from app.db.models.kyc_case import KycCase
from app.db.models.ocr_extraction import OcrExtraction
from app.db.models.user import User, UserRole
from app.db.session import get_db
from app.dependencies import get_current_user
from app.schemas.document import OcrExtractionResponse
from app.schemas.face_verification import FaceVerificationCaseSummary, FaceVerificationResultResponse
from app.schemas.explainability import GradCamSupplementaryResponse, TamperExplanationResponse

router = APIRouter(prefix="/verification", tags=["verification"])


def _assert_accessible(case: KycCase | None, case_id: uuid.UUID, current_user: User):
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    if current_user.role == UserRole.CUSTOMER and case.customer_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your case.")


@router.get("/{case_id}/ocr", response_model=list[OcrExtractionResponse])
def get_ocr_results(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = db.get(KycCase, case_id)
    _assert_accessible(case, case_id, current_user)
    docs = db.query(Document).filter(Document.kyc_case_id == case_id).all()
    results = []
    for doc in docs:
        if doc.ocr_extraction:
            results.append(doc.ocr_extraction)
    return results


@router.get("/{case_id}/tampering")
def get_tampering_results(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = db.get(KycCase, case_id)
    _assert_accessible(case, case_id, current_user)
    docs = db.query(Document).filter(Document.kyc_case_id == case_id).all()
    result = {}
    for doc in docs:
        if doc.tampering_result:
            key = doc.document_type.value.lower()
            result[key] = {
                "verdict": doc.tampering_result.verdict.value,
                "confidence": doc.tampering_result.confidence,
                "ela_score": doc.tampering_result.ela_score,
                "model_version": doc.tampering_result.model_version,
                "gradcam_heatmap_r2_key": doc.tampering_result.gradcam_heatmap_r2_key,
            }
    return result


@router.get("/{case_id}/face-match")
def get_face_match_results(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.db.models.face_verification_result import FaceVerificationResult
    from app.services.face_verification_service import case_overall_face_match

    case = db.get(KycCase, case_id)
    _assert_accessible(case, case_id, current_user)
    results = (
        db.query(FaceVerificationResult)
        .filter(FaceVerificationResult.kyc_case_id == case_id)
        .all()
    )
    if not results:
        raise HTTPException(status_code=404, detail="Face verification results not found.")
    overall = case_overall_face_match(results)
    avg_pct = sum(r.match_percentage for r in results) / len(results)
    return {
        "overall_match": overall,
        "average_match_pct": round(avg_pct, 2),
        "comparisons_run": len(results),
        "comparisons": [
            {
                "id_document_id": str(r.id_document_id),
                "is_match": r.is_match,
                "match_percentage": r.match_percentage,
                "cosine_similarity": r.cosine_similarity,
            }
            for r in results
        ],
    }


@router.get("/{case_id}/cross-document")
def get_cross_document_result(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.db.models.cross_document_result import CrossDocumentResult

    case = db.get(KycCase, case_id)
    _assert_accessible(case, case_id, current_user)
    result = (
        db.query(CrossDocumentResult)
        .filter(CrossDocumentResult.kyc_case_id == case_id)
        .one_or_none()
    )
    if not result:
        raise HTTPException(status_code=404, detail="Cross-document verification result not found.")
    return result


@router.get("/{case_id}/liveness")
def get_liveness_result(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.db.models.liveness_result import LivenessResult

    case = db.get(KycCase, case_id)
    _assert_accessible(case, case_id, current_user)
    result = (
        db.query(LivenessResult)
        .filter(LivenessResult.kyc_case_id == case_id)
        .one_or_none()
    )
    if not result:
        raise HTTPException(status_code=404, detail="Liveness result not found.")
    return result


@router.get("/{case_id}/gradcam")
def get_gradcam_result(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = db.get(KycCase, case_id)
    _assert_accessible(case, case_id, current_user)
    docs = db.query(Document).filter(Document.kyc_case_id == case_id).all()
    for doc in docs:
        if doc.tampering_result and doc.tampering_result.gradcam_heatmap_r2_key:
            return {
                "gradcam_heatmap_r2_key": doc.tampering_result.gradcam_heatmap_r2_key,
                "gradcam_heatmap_url": f"/api/v1/verification/{case_id}/gradcam/image",
                "predicted_imagenet_class_idx": None,
                "predicted_imagenet_class_confidence": None,
                "disclaimer": (
                    "This Grad-CAM visualization shows which regions most strongly activated "
                    "ResNet18's general visual features. It does NOT directly explain the "
                    "tampering verdict — see the primary explanation (feature attribution) "
                    "for the actual verdict rationale."
                ),
            }
    raise HTTPException(status_code=404, detail="No Grad-CAM heatmap available for this case.")
