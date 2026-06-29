"""
ResNet18 feature extractor (frozen, ImageNet-pretrained).

Per the locked decision: this is used PURELY as a frozen feature
extractor — no fine-tuning in this module. We take the output of the
penultimate layer (the 512-dim global-average-pooled activations right
before the final classification layer), which is the standard, documented
approach for using a pretrained CNN as a generic visual feature extractor
for a downstream classical-ML classifier (Random Forest).

Why this is legitimate without forgery-specific training data: ImageNet
pretraining gives ResNet18 general-purpose texture/edge/color statistical
sensitivity. Tampered regions (splices, copy-move, font overlays) often
introduce subtle texture/noise/edge discontinuities that these generic
features are known to respond to, even without forgery-specific
fine-tuning. This is a documented technique in forgery-detection
literature, not a fabricated capability — but it is acknowledged here
explicitly as a heuristic feature source, with the Random Forest providing
the actual forgery-specific decision boundary once trained on real
labeled data.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_feature_extractor: nn.Module | None = None
_preprocess: transforms.Compose | None = None

RESNET_FEATURE_DIM = 512


def _build_feature_extractor() -> nn.Module:
    """
    Load ImageNet-pretrained ResNet18 and strip the final fully-connected
    classification layer, exposing the 512-dim pooled feature vector.
    Set to eval() and all parameters frozen (requires_grad=False) since
    this module performs no training/fine-tuning.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])  # drop final fc layer
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_feature_extractor() -> tuple[nn.Module, transforms.Compose]:
    """
    Lazily initialize and return the singleton (model, preprocess_transform)
    pair. Thread-safe for concurrent Celery worker access.
    """
    global _feature_extractor, _preprocess

    if _feature_extractor is not None and _preprocess is not None:
        return _feature_extractor, _preprocess

    with _model_lock:
        if _feature_extractor is None:
            logger.info("Loading ResNet18 (ImageNet-pretrained) feature extractor...")
            _feature_extractor = _build_feature_extractor()
            weights = ResNet18_Weights.IMAGENET1K_V1
            _preprocess = weights.transforms()
            logger.info("ResNet18 feature extractor loaded.")

    return _feature_extractor, _preprocess


def extract_resnet_features(pil_image: Image.Image) -> np.ndarray:
    """
    Run the frozen ResNet18 feature extractor on a PIL image and return a
    flat 512-dim numpy feature vector.

    Args:
        pil_image: RGB PIL Image (any size — the official ResNet18 weights
            transform handles resizing/normalization internally)

    Returns:
        np.ndarray of shape (512,), dtype float32
    """
    model, preprocess = get_feature_extractor()

    input_tensor = preprocess(pil_image).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        output = model(input_tensor)  # (1, 512, 1, 1)

    feature_vector = output.squeeze().numpy().astype(np.float32)  # (512,)

    if feature_vector.shape != (RESNET_FEATURE_DIM,):
        raise RuntimeError(
            f"Unexpected ResNet18 feature vector shape: {feature_vector.shape}, "
            f"expected ({RESNET_FEATURE_DIM},)"
        )

    return feature_vector


def feature_names() -> list[str]:
    """Generic positional names for the 512 ResNet18 features (used for
    feature-importance reporting in explainability, not semantically
    meaningful individually)."""
    return [f"resnet_feat_{i}" for i in range(RESNET_FEATURE_DIM)]
