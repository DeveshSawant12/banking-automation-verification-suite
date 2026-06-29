"""
Error Level Analysis (ELA).

How ELA works (standard, well-documented forensic technique — not invented
here): a JPEG image is re-saved at a known quality level, then the
pixel-wise absolute difference between the original and the re-saved
version is computed. Regions of an image that were edited/spliced after
the original compression will have a DIFFERENT error level than the rest
of the image, because they went through a different compression history.
Untampered regions converge to a low, uniform error level after a single
re-compression; tampered/spliced regions stand out as bright patches in
the ELA output.

This module produces:
  1. A visual ELA difference image (numpy array) — used later by
     Grad-CAM/visualization (Module 8) to highlight suspicious regions.
  2. A compact set of numeric ELA features (mean, std, max, high-error
     pixel ratio, regional variance) — used as input to the Random Forest
     classifier in this module.

No external dataset or pretrained model is required for ELA itself; it is
a deterministic image-processing technique.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

ELA_JPEG_QUALITY = 90
ELA_SCALE_FACTOR = 15  # standard ELA brightness amplification for visualization


@dataclass
class ElaFeatures:
    mean_error: float
    std_error: float
    max_error: float
    high_error_pixel_ratio: float  # fraction of pixels above a high-error threshold
    regional_variance: float  # variance of block-wise mean errors (splice indicator)

    def to_vector(self) -> np.ndarray:
        """Return as a fixed-order numpy feature vector for the RF classifier."""
        return np.array(
            [
                self.mean_error,
                self.std_error,
                self.max_error,
                self.high_error_pixel_ratio,
                self.regional_variance,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "ela_mean_error",
            "ela_std_error",
            "ela_max_error",
            "ela_high_error_pixel_ratio",
            "ela_regional_variance",
        ]


def generate_ela_image(
    pil_image: Image.Image, jpeg_quality: int = ELA_JPEG_QUALITY
) -> np.ndarray:
    """
    Generate the ELA difference image.

    Args:
        pil_image: original image as PIL Image (RGB)
        jpeg_quality: re-compression quality level (90 is the standard
            default used in forensic ELA literature — high enough to avoid
            introducing excessive new artifacts, low enough to reveal
            recompression differences)

    Returns:
        ELA difference image as a uint8 numpy array (H, W, 3), scaled for
        visibility. Pixel intensity correlates with local error level.
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, "JPEG", quality=jpeg_quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    original_arr = np.array(pil_image).astype(np.int16)
    recompressed_arr = np.array(recompressed).astype(np.int16)

    if original_arr.shape != recompressed_arr.shape:
        # Defensive: JPEG re-encoding should preserve dimensions, but if a
        # mode conversion altered channel count, align explicitly.
        recompressed = recompressed.resize(pil_image.size)
        recompressed_arr = np.array(recompressed).astype(np.int16)

    diff = np.abs(original_arr - recompressed_arr)
    ela_image = (diff * ELA_SCALE_FACTOR).clip(0, 255).astype(np.uint8)
    return ela_image


def extract_ela_features(ela_image: np.ndarray) -> ElaFeatures:
    """
    Compute numeric ELA features from the ELA difference image for use as
    Random Forest input.

    Regional variance is computed by splitting the image into an 8x8 grid
    of blocks and measuring the variance of per-block mean error — a
    spliced/tampered region produces a sharp local spike that raises this
    variance relative to a uniformly-compressed authentic image.
    """
    gray_ela = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY)

    mean_error = float(np.mean(gray_ela))
    std_error = float(np.std(gray_ela))
    max_error = float(np.max(gray_ela))

    high_error_threshold = 50  # empirically standard threshold in ELA literature
    high_error_pixel_ratio = float(
        np.sum(gray_ela > high_error_threshold) / gray_ela.size
    )

    h, w = gray_ela.shape
    grid_size = 8
    block_h, block_w = max(h // grid_size, 1), max(w // grid_size, 1)
    block_means = []
    for i in range(0, h - block_h + 1, block_h):
        for j in range(0, w - block_w + 1, block_w):
            block = gray_ela[i : i + block_h, j : j + block_w]
            block_means.append(np.mean(block))
    regional_variance = float(np.var(block_means)) if block_means else 0.0

    return ElaFeatures(
        mean_error=mean_error,
        std_error=std_error,
        max_error=max_error,
        high_error_pixel_ratio=high_error_pixel_ratio,
        regional_variance=regional_variance,
    )


def run_ela_pipeline(pil_image: Image.Image) -> tuple[np.ndarray, ElaFeatures]:
    """Convenience entry point: image -> (ela_visual_image, ela_features)."""
    ela_image = generate_ela_image(pil_image)
    features = extract_ela_features(ela_image)
    return ela_image, features
