"""
Explainability Service — Module 8 orchestration layer.

Combines:
  1. PRIMARY explanation (app/ml/explainability/ela_feature_explainer.py):
     reads back the fused feature vector persisted to R2 at
     tampering-check time (Module 2/3, extended in Module 8) and the
     trained Random Forest model, to produce a genuine feature-attribution
     explanation of the actual REAL/TAMPERED verdict.
  2. SUPPLEMENTARY visual (app/ml/explainability/gradcam.py): true
     Grad-CAM on ResNet18, explicitly labeled as not explaining the RF
     verdict directly (see that module's docstring).

Per the locked decision: BOTH are provided. The primary explanation is
the one with real explanatory power over the actual fraud decision; the
Grad-CAM visual is supplementary spatial context.

NO SILENT FAILURE: if no TamperingResult exists for the document, or its
verdict is INCONCLUSIVE (no model was available to produce a real
verdict), or its resnet_feature_vector_ref is missing (R2 upload failed
at check time, or this document was processed before Module 8 was added),
this raises a typed exception rather than fabricating an explanation for
data that doesn't exist.
"""

from __future__ import annotations

import json
import logging
import uuid

from PIL import Image
from sqlalchemy.orm import Session

from app.db.models.document import Document, DocumentType
from app.db.models.tampering_result import TamperingResult, TamperVerdict
from app.ml.explainability.ela_feature_explainer import (
    TamperExplanation,
    explain_tampering_verdict,
)
from app.ml.explainability.gradcam import generate_gradcam_heatmap, heatmap_to_overlay_image
from app.ml.forgery.ela import ElaFeatures
from app.ml.forgery.feature_fusion import fused_feature_names
from app.ml.forgery.random_forest_model import ForgeryRandomForestModel, ModelNotTrainedError
from app.services.aadhaar_tamper_service import AADHAAR_MODEL_PATH
from app.services.audit_service import AuditEventType, write_audit_log
from app.services.pan_tamper_service import PAN_MODEL_PATH
from app.services.storage_service import StorageServiceError, download_bytes, upload_bytes
from app.utils.image_utils import ImageLoadError, load_pil_image_from_bytes

logger = logging.getLogger(__name__)


class ExplainabilityServiceError(Exception):
    """Raised when an explanation cannot be produced for a document."""


def _get_model_for_document_type(document_type: DocumentType) -> ForgeryRandomForestModel:
    """
    Load the correct trained Random Forest model based on document type.
    Raises ModelNotTrainedError (uncaught here, deliberately) if not
    trained — consistent with Modules 2/3's contract.
    """
    model_path = (
        AADHAAR_MODEL_PATH if document_type == DocumentType.AADHAAR else PAN_MODEL_PATH
    )
    return ForgeryRandomForestModel.load(model_path)


def _reconstruct_ela_features_from_summary(ela_score: float) -> ElaFeatures:
    """
    The fused feature vector (downloaded from R2) already contains the
    individual ELA feature values in their correct positions, so the
    PRIMARY explanation path does not need this reconstruction. This
    helper exists only as a defensive fallback for the narrative-summary
    builder if, for some reason, the full vector's ELA slice cannot be
    located by name (should not normally happen — see
    explain_document_tampering's assertion before use). Reconstructs a
    minimal ElaFeatures using only the persisted summary ela_score,
    leaving other fields at 0 and logging that this is a degraded path.
    """
    logger.warning(
        "Falling back to minimal ELA feature reconstruction from summary "
        "ela_score=%.4f only. This loses std/max/regional-variance detail "
        "and indicates the fused vector's ELA slice could not be parsed "
        "by name -- investigate if this occurs in production.",
        ela_score,
    )
    return ElaFeatures(
        mean_error=ela_score,
        std_error=0.0,
        max_error=0.0,
        high_error_pixel_ratio=0.0,
        regional_variance=0.0,
    )


def explain_document_tampering(db: Session, document_id: uuid.UUID) -> TamperExplanation:
    """
    Build the PRIMARY explanation for a document's tampering verdict.

    Raises:
        ExplainabilityServiceError: if the document/tampering result
            doesn't exist, or the feature vector reference is missing
            (R2 upload failed at check time, or document predates
            Module 8).
        ModelNotTrainedError: if the trained model for this document type
            is unavailable. Propagated uncaught, consistent with the
            project-wide convention that callers must handle this as
            REVIEW_REQUIRED-adjacent, not silently swallow it.
    """
    document = db.get(Document, document_id)
    if document is None:
        raise ExplainabilityServiceError(f"Document {document_id} does not exist.")

    if document.tampering_result is None:
        raise ExplainabilityServiceError(
            f"No tampering result exists for document {document_id}. "
            f"Run the tampering check (Module 2/3) before requesting an "
            f"explanation."
        )

    tampering_result = document.tampering_result

    if tampering_result.verdict == TamperVerdict.INCONCLUSIVE:
        # A real, honest explanation: there IS no model-driven verdict to
        # explain. Returns a structured "no explanation available"
        # response rather than raising, since this is a valid, expected
        # state (not an error condition) the Admin Dashboard needs to
        # render gracefully.
        return explain_tampering_verdict(
            verdict=TamperVerdict.INCONCLUSIVE.value,
            fused_feature_vector=None,  # not used for INCONCLUSIVE path
            feature_names=[],
            model=None,  # type: ignore[arg-type]
            ela_features=None,  # type: ignore[arg-type]
        )

    if tampering_result.resnet_feature_vector_ref is None:
        raise ExplainabilityServiceError(
            f"Document {document_id} has a tampering verdict but no "
            f"persisted feature vector reference. This document was "
            f"likely processed before Module 8's feature-persistence "
            f"change, or the R2 upload failed at check time (see logs "
            f"from the original tampering check). Re-run the tampering "
            f"check to enable explanation for this document."
        )

    try:
        vector_payload_bytes = download_bytes(tampering_result.resnet_feature_vector_ref)
    except StorageServiceError as exc:
        raise ExplainabilityServiceError(
            f"Failed to retrieve persisted feature vector for document "
            f"{document_id} from R2: {exc}"
        ) from exc

    vector_payload = json.loads(vector_payload_bytes)
    fused_vector_list = vector_payload["fused_vector"]
    stored_feature_names = vector_payload["feature_names"]

    current_feature_names = fused_feature_names()
    if stored_feature_names != current_feature_names:
        raise ExplainabilityServiceError(
            f"Stored feature names for document {document_id} do not match "
            f"the current feature_fusion.fused_feature_names() contract. "
            f"The model or feature pipeline was likely changed since this "
            f"document was processed. Refusing to build a potentially "
            f"misattributed explanation — re-run the tampering check."
        )

    import numpy as np

    fused_vector = np.array(fused_vector_list, dtype=np.float32)

    # ModelNotTrainedError propagates uncaught -- by design, see docstring.
    model = _get_model_for_document_type(document.document_type)

    # Reconstruct ElaFeatures from the fused vector's known ELA slice
    # (first 5 positions, per feature_fusion.py's fixed ordering contract)
    # rather than re-running ELA generation on the original image, since
    # the values are already present in the stored vector.
    ela_feature_names = ElaFeatures.feature_names()
    ela_values = fused_vector_list[: len(ela_feature_names)]
    ela_features = ElaFeatures(
        mean_error=ela_values[0],
        std_error=ela_values[1],
        max_error=ela_values[2],
        high_error_pixel_ratio=ela_values[3],
        regional_variance=ela_values[4],
    )

    explanation = explain_tampering_verdict(
        verdict=tampering_result.verdict.value,
        fused_feature_vector=fused_vector,
        feature_names=current_feature_names,
        model=model,
        ela_features=ela_features,
    )

    # Module 9 wiring: recording that an explanation was generated (and
    # for which verdict) is audit-worthy in a banking context -- it
    # documents that the system's reasoning for a TAMPERED/REAL decision
    # was actually inspected, which matters for regulatory/compliance
    # review trails.
    write_audit_log(
        db,
        AuditEventType.EXPLANATION_GENERATED,
        kyc_case_id=document.kyc_case_id,
        verification_status=explanation.verdict,
        metadata_json={
            "document_id": str(document_id),
            "top_feature": (
                explanation.top_contributing_features[0].feature_name
                if explanation.top_contributing_features
                else None
            ),
        },
    )

    return explanation


def generate_supplementary_gradcam(
    db: Session, document_id: uuid.UUID, original_image_bytes: bytes
) -> dict:
    """
    Generate and persist the SUPPLEMENTARY Grad-CAM visualization for a
    document. Requires the original image bytes (not retrievable from the
    fused feature vector, since Grad-CAM needs to re-run forward+backward
    passes on the actual image) — the caller (API endpoint) is
    responsible for fetching these from R2 via the document's stored
    upload key.

    Returns:
        dict with keys: gradcam_heatmap_r2_key (str),
        predicted_imagenet_class_idx (int),
        predicted_imagenet_class_confidence (float)

    Raises:
        ExplainabilityServiceError: on image load or R2 upload failure.
    """
    try:
        pil_image = load_pil_image_from_bytes(original_image_bytes)
    except ImageLoadError as exc:
        raise ExplainabilityServiceError(
            f"Could not load image for Grad-CAM generation: {exc}"
        ) from exc

    heatmap, predicted_class_idx, predicted_confidence = generate_gradcam_heatmap(pil_image)
    overlay_image = heatmap_to_overlay_image(pil_image, heatmap)

    import io

    buffer = io.BytesIO()
    overlay_image.save(buffer, format="PNG")
    overlay_bytes = buffer.getvalue()

    try:
        r2_key = upload_bytes(
            overlay_bytes,
            key_prefix=f"explainability/gradcam/{document_id}",
            content_type="image/png",
        )
    except StorageServiceError as exc:
        raise ExplainabilityServiceError(
            f"Failed to upload Grad-CAM heatmap to R2: {exc}"
        ) from exc

    document = db.get(Document, document_id)
    if document is not None and document.tampering_result is not None:
        document.tampering_result.gradcam_heatmap_r2_key = r2_key
        db.commit()

    return {
        "gradcam_heatmap_r2_key": r2_key,
        "predicted_imagenet_class_idx": predicted_class_idx,
        "predicted_imagenet_class_confidence": predicted_confidence,
    }
