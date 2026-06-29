"""
EasyOCR engine wrapper.

EasyOCR is loaded once per process (model weights are large; re-instantiating
per-request would be extremely slow and memory-wasteful). The reader is
configured for English + Hindi since Aadhaar cards print both languages,
which is the real, documented bilingual format of the card — not an
invented requirement.

This module ONLY handles raw text detection/recognition. Field-level
parsing (name/DOB/gender/address/number extraction) is handled by
ocr_service.py using regex_patterns.py, keeping this module a thin,
testable wrapper around the EasyOCR library itself.
"""

from __future__ import annotations

import logging
import threading

import easyocr
import numpy as np

logger = logging.getLogger(__name__)

_reader_lock = threading.Lock()
_reader_instance: easyocr.Reader | None = None


def get_ocr_reader() -> easyocr.Reader:
    """
    Lazily initialize and return a singleton EasyOCR Reader instance.
    Thread-safe via lock to avoid duplicate initialization under concurrent
    Celery worker startup.
    """
    global _reader_instance

    if _reader_instance is not None:
        return _reader_instance

    with _reader_lock:
        if _reader_instance is None:
            logger.info("Initializing EasyOCR reader (en, hi)...")
            _reader_instance = easyocr.Reader(
                ["en", "hi"],
                gpu=False,
                verbose=False,
                model_storage_directory="/app/ml_models/easyocr",
            )   
            logger.info("EasyOCR reader initialized.")
    return _reader_instance


class OcrResult:
    """
    Structured container for a single EasyOCR detection: bounding box,
    recognized text, and confidence score.
    """

    __slots__ = ("bbox", "text", "confidence")

    def __init__(self, bbox: list, text: str, confidence: float):
        self.bbox = bbox
        self.text = text
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
        }


def run_ocr(image: np.ndarray, min_confidence: float = 0.25) -> list[OcrResult]:
    """
    Run EasyOCR text detection + recognition on a preprocessed image array.

    Args:
        image: BGR numpy array (output of image_utils.preprocess_for_ocr)
        min_confidence: detections below this confidence are discarded as
            noise (EasyOCR frequently emits very low-confidence junk boxes
            on document backgrounds/watermarks)

    Returns:
        List of OcrResult objects, ordered as returned by EasyOCR
        (top-to-bottom, left-to-right approximate reading order).

    Raises:
        RuntimeError: if EasyOCR produces no detections at all, which
            indicates either a blank/corrupted image or a fundamentally
            unreadable upload — the caller must surface this as a
            verification failure, not silently proceed.
    """
    reader = get_ocr_reader()
    raw_results = reader.readtext(image)

    if not raw_results:
        raise RuntimeError(
            "EasyOCR returned zero detections. Image may be blank, "
            "unreadable, or not a document."
        )

    filtered: list[OcrResult] = []
    for bbox, text, confidence in raw_results:
        if confidence < min_confidence:
            continue
        # EasyOCR returns bbox corners as numpy types; cast to plain floats
        # so this is JSON-serializable downstream (raw_ocr_json column).
        clean_bbox = [[float(x), float(y)] for x, y in bbox]
        filtered.append(OcrResult(clean_bbox, text.strip(), confidence))

    if not filtered:
        raise RuntimeError(
            f"All EasyOCR detections fell below min_confidence={min_confidence}. "
            "Image quality is likely too poor for reliable extraction."
        )

    return filtered


def get_full_text(results: list[OcrResult]) -> str:
    """Concatenate all detected text fragments into a single block for regex parsing."""
    return "\n".join(r.text for r in results)
