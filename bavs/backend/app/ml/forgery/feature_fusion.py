"""
Feature Fusion.

Per the locked decision: fixed concatenation of all three feature groups,
in a fixed, documented order:

    [ ELA features (5) | ResNet18 features (512) | OCR-derived features (6) ]
    Total dimensionality: 523

This fixed order is REQUIRED to be identical between training
(train_forgery_model.py) and inference (aadhaar_tamper_service.py /
pan_tamper_service.py) — any mismatch would silently corrupt predictions.
This module is the single source of truth for that order so both training
and inference import from here rather than duplicating the concatenation
logic.
"""

from __future__ import annotations

import numpy as np

from app.ml.forgery.ela import ElaFeatures
from app.ml.forgery.ocr_derived_features import OcrDerivedFeatures

FEATURE_GROUPS = ("ela", "resnet18", "ocr")


def fuse_features(
    ela_features: ElaFeatures,
    resnet_features: np.ndarray,
    ocr_features: OcrDerivedFeatures,
) -> np.ndarray:
    """
    Concatenate ELA + ResNet18 + OCR-derived features into a single fixed
    fusion vector.

    Args:
        ela_features: output of ela.extract_ela_features
        resnet_features: output of resnet_feature_extractor.extract_resnet_features,
            shape (512,)
        ocr_features: output of ocr_derived_features.extract_ocr_derived_features

    Returns:
        np.ndarray of shape (523,), dtype float32

    Raises:
        ValueError: if resnet_features is not the expected shape, which
            would indicate a caller bug upstream (never silently pad/truncate
            a feature vector feeding a fraud decision).
    """
    if resnet_features.shape != (512,):
        raise ValueError(
            f"Expected ResNet18 feature vector of shape (512,), "
            f"got {resnet_features.shape}. Refusing to fuse — this would "
            f"silently corrupt the Random Forest input."
        )

    ela_vec = ela_features.to_vector()
    ocr_vec = ocr_features.to_vector()

    fused = np.concatenate([ela_vec, resnet_features, ocr_vec]).astype(np.float32)

    expected_dim = len(ElaFeatures.feature_names()) + 512 + len(
        OcrDerivedFeatures.feature_names()
    )
    if fused.shape[0] != expected_dim:
        raise ValueError(
            f"Fused feature vector has unexpected dimensionality "
            f"{fused.shape[0]}, expected {expected_dim}."
        )

    return fused


def fused_feature_names() -> list[str]:
    """
    Full ordered list of feature names corresponding to fuse_features()
    output. Used for Random Forest feature_importances_ reporting in the
    Explainable AI module (Module 8).
    """
    return (
        ElaFeatures.feature_names()
        + [f"resnet_feat_{i}" for i in range(512)]
        + OcrDerivedFeatures.feature_names()
    )


def fused_vector_dimensionality() -> int:
    return len(fused_feature_names())
