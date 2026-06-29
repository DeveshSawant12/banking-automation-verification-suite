"""
Aadhaar Tampering Detection Service — Module 2 orchestration layer.

Pipeline (per locked spec):
    Aadhaar Image -> Preprocessing -> ELA Generation -> Feature Extraction
    (OCR + ELA + ResNet18) -> Feature Fusion -> Random Forest ->
    REAL or TAMPERED

This service composes the independently-tested components from
app/ml/forgery/*.py and app/ml/ocr/*.py. It does NOT duplicate their
logic. Like ocr_service.py, this is a plain callable (not a Celery task
itself) so it can be wrapped by the pipeline orchestrator later without
duplicating logic, and can be called directly from an endpoint for
single-document testing.

CRITICAL SAFETY BEHAVIOR: if no trained Random Forest model exists for
Aadhaar at the configured path, this service does NOT fabricate a
verdict. It persists a TamperingResult row with verdict=INCONCLUSIVE and
re-raises information the caller needs to route the case to
REVIEW_REQUIRED. This mirrors the ModelNotTrainedError contract from
random_forest_model.py.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from app.db.models.document import Document
from app.db.models.tampering_result import TamperingResult, TamperVerdict
from app.ml.forgery.ela import run_ela_pipeline
from app.ml.forgery.feature_fusion import fuse_features, fused_feature_names
from app.ml.forgery.ocr_derived_features import extract_ocr_derived_features
from app.ml.forgery.random_forest_model import (
    ForgeryRandomForestModel,
    ModelNotTrainedError,
)
from app.ml.forgery.resnet_feature_extractor import extract_resnet_features
from app.ml.ocr.easyocr_engine import run_ocr
from app.ml.ocr.field_parser import parse_fields
from app.services.audit_service import AuditEventType, write_audit_log
from app.services.storage_service import StorageServiceError, upload_bytes
from app.utils.image_utils import ImageLoadError, load_pil_image_from_bytes, preprocess_for_ocr

logger = logging.getLogger(__name__)

AADHAAR_MODEL_PATH = Path("ml_models/aadhaar_rf_model.pkl")

_cached_model: ForgeryRandomForestModel | None = None


class AadhaarTamperingServiceError(Exception):
    """Raised on unrecoverable errors during Aadhaar tampering detection."""


def _get_aadhaar_model() -> ForgeryRandomForestModel:
    """
    Lazily load and cache the Aadhaar Random Forest model from disk.
    Raises ModelNotTrainedError (uncaught here, deliberately — the caller
    decides how to handle it) if no trained model exists yet.
    """
    global _cached_model
    if _cached_model is None:
        _cached_model = ForgeryRandomForestModel.load(AADHAAR_MODEL_PATH)
    return _cached_model


def analyze_aadhaar_tampering(file_bytes: bytes) -> dict:
    """
    Run the full Aadhaar tampering detection pipeline on raw image bytes.

    Returns:
        dict with keys: verdict (str), confidence (float), ela_score (float),
        model_version (str)

    Raises:
        AadhaarTamperingServiceError: on image loading or OCR/ELA/ResNet
            extraction failure.
        ModelNotTrainedError: if no trained model exists. Callers MUST
            catch this and treat the case as REVIEW_REQUIRED, never as an
            implicit verdict.
    """
    try:
        pil_image = load_pil_image_from_bytes(file_bytes)
    except ImageLoadError as exc:
        raise AadhaarTamperingServiceError(f"Could not load image: {exc}") from exc

    try:
        ocr_input_image = preprocess_for_ocr(file_bytes)
        ocr_results = run_ocr(ocr_input_image)
    except (ImageLoadError, RuntimeError) as exc:
        raise AadhaarTamperingServiceError(
            f"OCR extraction failed during tampering analysis: {exc}"
        ) from exc

    parsed_fields = parse_fields(ocr_results)
    ocr_features = extract_ocr_derived_features(ocr_results, parsed_fields)

    ela_image, ela_features = run_ela_pipeline(pil_image)
    resnet_features = extract_resnet_features(pil_image)

    fused_vector = fuse_features(ela_features, resnet_features, ocr_features)

    # ModelNotTrainedError propagates uncaught -- by design, see docstring.
    model = _get_aadhaar_model()
    prediction = model.predict(fused_vector)

    return {
        "verdict": prediction.verdict,
        "confidence": prediction.confidence,
        "ela_score": ela_features.mean_error,
        "model_version": prediction.model_version,
        "ela_image": ela_image,  # passed through for Grad-CAM/visualization (Module 8)
        "fused_vector": fused_vector,  # added in Module 8: persisted to R2 for explainability
    }


def persist_tampering_result(
    db: Session,
    document_id: uuid.UUID,
    verdict: str,
    confidence: float,
    ela_score: float,
    model_version: str,
    gradcam_heatmap_r2_key: str | None = None,
    resnet_feature_vector_ref: str | None = None,
) -> TamperingResult:
    """
    Persist a tampering analysis result. verdict must be one of
    TamperVerdict's values ("REAL", "TAMPERED", "INCONCLUSIVE").

    resnet_feature_vector_ref (added in Module 8): R2 object key pointing
    to the full 523-dim fused feature vector (ELA + ResNet18 + OCR),
    stored as JSON. Despite the column's legacy name (it predates feature
    fusion being finalized), it stores the FULL fused vector, not just
    ResNet18's 512 dims, since that's what Module 8's explainability
    service needs to reconstruct feature-level attribution without
    re-running the entire OCR/ELA/ResNet pipeline on every explanation
    request.
    """
    document = db.get(Document, document_id)
    if document is None:
        raise AadhaarTamperingServiceError(f"Document {document_id} does not exist.")

    existing = (
        db.query(TamperingResult)
        .filter(TamperingResult.document_id == document_id)
        .one_or_none()
    )
    if existing is not None:
        raise AadhaarTamperingServiceError(
            f"Tampering result already exists for document {document_id}."
        )

    result = TamperingResult(
        document_id=document_id,
        verdict=TamperVerdict(verdict),
        confidence=confidence,
        ela_score=ela_score,
        model_version=model_version,
        gradcam_heatmap_r2_key=gradcam_heatmap_r2_key,
        resnet_feature_vector_ref=resnet_feature_vector_ref,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def run_and_persist_aadhaar_tampering_check(
    db: Session, document_id: uuid.UUID, file_bytes: bytes
) -> TamperingResult:
    """
    Convenience wrapper: runs the full pipeline and persists the result.

    If no trained model exists, persists an INCONCLUSIVE verdict (rather
    than crashing the whole pipeline run or silently skipping the row)
    and logs a clear warning. The orchestrator (built once all modules
    exist) is responsible for routing INCONCLUSIVE tampering results to
    REVIEW_REQUIRED case status.
    """
    try:
        analysis = analyze_aadhaar_tampering(file_bytes)

        feature_vector_ref = None
        try:
            vector_payload = json.dumps(
                {
                    "fused_vector": analysis["fused_vector"].tolist(),
                    "feature_names": fused_feature_names(),
                }
            ).encode("utf-8")
            feature_vector_ref = upload_bytes(
                vector_payload,
                key_prefix=f"explainability/feature_vectors/aadhaar/{document_id}",
                content_type="application/json",
            )
        except StorageServiceError as exc:
            # The tampering verdict itself is still valid and gets persisted
            # below regardless -- losing the explainability artifact is a
            # degraded-but-non-fatal condition, not a reason to fail the
            # whole tampering check. Logged clearly so it's not silent.
            logger.warning(
                "Failed to persist feature vector to R2 for document %s "
                "(tampering verdict will still be saved, but Module 8 "
                "explanation will be unavailable for this document): %s",
                document_id,
                exc,
            )

        result = persist_tampering_result(
            db,
            document_id,
            verdict=analysis["verdict"],
            confidence=analysis["confidence"],
            ela_score=analysis["ela_score"],
            model_version=analysis["model_version"],
            resnet_feature_vector_ref=feature_vector_ref,
        )
    except ModelNotTrainedError as exc:
        logger.warning(
            "No trained Aadhaar tampering model available for document %s: %s. "
            "Persisting INCONCLUSIVE verdict; case must be routed to "
            "REVIEW_REQUIRED by the orchestrator.",
            document_id,
            exc,
        )
        # ELA score is still meaningful without a trained RF model, but to
        # avoid duplicating the full pipeline here, we recompute only what's
        # cheap and safe: re-run ELA alone for a partial signal.
        try:
            pil_image = load_pil_image_from_bytes(file_bytes)
            _, ela_features = run_ela_pipeline(pil_image)
            ela_score = ela_features.mean_error
        except ImageLoadError:
            ela_score = 0.0

        result = persist_tampering_result(
            db,
            document_id,
            verdict=TamperVerdict.INCONCLUSIVE.value,
            confidence=0.0,
            ela_score=ela_score,
            model_version="NO_MODEL_TRAINED",
        )

    # Module 9 wiring: single audit-write code path covers both the
    # success and INCONCLUSIVE-fallback branches above, since `result`
    # is set by either before reaching here.
    document = db.get(Document, document_id)
    write_audit_log(
        db,
        AuditEventType.AADHAAR_TAMPERING_CHECK_COMPLETED,
        kyc_case_id=document.kyc_case_id if document else None,
        verification_status=result.verdict.value,
        metadata_json={
            "document_id": str(document_id),
            "confidence": result.confidence,
            "model_version": result.model_version,
        },
    )

    return result
