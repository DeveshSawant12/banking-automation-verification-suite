"""
Shared image preprocessing utilities used by OCR and forgery-detection pipelines.

These operations are deliberately generic (not Aadhaar/PAN-specific) so they
can be reused by Module 1 (OCR) and Module 2/3 (tampering detection) without
duplication.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image


class ImageLoadError(Exception):
    """Raised when an uploaded file cannot be decoded as a valid image."""


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Decode raw uploaded bytes into a BGR numpy array (OpenCV convention).
    Raises ImageLoadError if the bytes do not represent a valid image.
    """
    if not file_bytes:
        raise ImageLoadError("Empty file content provided.")

    np_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ImageLoadError(
            "Could not decode image. File may be corrupted or in an "
            "unsupported format."
        )
    return image


def load_pil_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """
    Decode raw uploaded bytes into a PIL Image (RGB). Used specifically by
    ELA, which requires PIL's JPEG re-encoding behavior.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = image.convert("RGB")
        return image
    except Exception as exc:
        raise ImageLoadError(f"Could not decode image with PIL: {exc}") from exc


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew in scanned/photographed ID documents using
    minAreaRect on thresholded text regions. Improves OCR accuracy on
    angled phone-camera captures.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        # Not enough foreground pixels to estimate a reliable angle.
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip correction for negligible skew to avoid introducing artifacts.
    if abs(angle) < 0.5:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def denoise_and_sharpen(image: np.ndarray) -> np.ndarray:
    """
    Apply mild denoising followed by unsharp masking to improve text
    legibility for OCR on low-quality phone camera captures.
    """
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
    gaussian = cv2.GaussianBlur(denoised, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    return sharpened


def resize_for_ocr(image: np.ndarray, target_width: int = 1600) -> np.ndarray:
    """
    Upscale small images to a target width to improve EasyOCR recognition
    accuracy on low-resolution uploads, while leaving larger images
    untouched to control inference time.
    """
    height, width = image.shape[:2]
    if width >= target_width:
        return image
    scale = target_width / width
    new_size = (target_width, int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def preprocess_for_ocr(file_bytes: bytes) -> np.ndarray:
    """
    Full preprocessing chain applied before handing an image to EasyOCR:
    decode -> resize -> deskew -> denoise/sharpen.
    """
    image = load_image_from_bytes(file_bytes)
    image = resize_for_ocr(image)
    image = deskew_image(image)
    image = denoise_and_sharpen(image)
    return image


def compute_sha256(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of file bytes for tamper-evidence and dedup."""
    import hashlib

    return hashlib.sha256(file_bytes).hexdigest()
