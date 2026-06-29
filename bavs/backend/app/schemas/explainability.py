"""
Pydantic schemas for Module 8 (Explainable AI).
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel


class FeatureContributionResponse(BaseModel):
    feature_name: str
    feature_value: float
    model_importance_rank: int
    model_importance_score: float


class TamperExplanationResponse(BaseModel):
    """
    Primary explanation response — directly explains the Random Forest's
    REAL/TAMPERED verdict via feature attribution (ela_feature_explainer.py).
    """

    document_id: uuid.UUID
    verdict: str
    top_contributing_features: list[FeatureContributionResponse]
    ela_feature_group_summary: dict
    resnet_feature_group_total_importance: float
    ocr_feature_group_summary: dict
    narrative_summary: str
    ela_heatmap_r2_key: str | None = None


class GradCamSupplementaryResponse(BaseModel):
    """
    Supplementary visual response — genuine Grad-CAM on ResNet18's
    ImageNet predictions, explicitly labeled as NOT explaining the actual
    tamper verdict (see gradcam.py module docstring for full rationale).
    """

    document_id: uuid.UUID
    gradcam_heatmap_r2_key: str
    predicted_imagenet_class_idx: int
    predicted_imagenet_class_confidence: float
    disclaimer: str = (
        "This Grad-CAM visualization shows which regions most strongly "
        "activated the ResNet18 model's general visual features. It does "
        "NOT directly explain the tampering verdict, which is produced by "
        "a separate Random Forest classifier. See the primary explanation "
        "(top_contributing_features) for the actual verdict rationale."
    )
