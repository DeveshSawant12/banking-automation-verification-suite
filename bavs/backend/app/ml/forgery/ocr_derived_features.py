"""
OCR-derived numeric features for the forgery-detection feature fusion
vector (the third feature group alongside ELA and ResNet18, per Module 2
spec: "OCR Features").

Rationale (not invented — this is a standard signal used in document
forgery literature): tampered text regions (e.g. a digit altered in an
Aadhaar/PAN number, or a name retyped over the original) frequently
produce LOWER OCR confidence than genuine printed text, because edited
regions often have inconsistent font rendering, anti-aliasing, or subtle
pixel-level artifacts from the editing tool. We also check whether the
extracted Aadhaar/PAN number passes structural/checksum-style format
validation (app.utils.regex_patterns) — a document where the visible
number fails format validation is itself a meaningful signal, independent
of ELA/ResNet.

This module consumes the OcrResult list (Module 1 output) and the parsed
fields (Module 1 field_parser output) — it does NOT re-run OCR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.ocr.easyocr_engine import OcrResult
from app.ml.ocr.field_parser import ParsedDocumentFields
from app.utils.regex_patterns import is_valid_aadhaar_format, is_valid_pan_format


@dataclass
class OcrDerivedFeatures:
    mean_confidence: float
    min_confidence: float
    std_confidence: float
    low_confidence_ratio: float  # fraction of detections below threshold
    id_number_format_valid: float  # 1.0 valid, 0.0 invalid/missing
    field_extraction_completeness: float  # fraction of required fields found

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.mean_confidence,
                self.min_confidence,
                self.std_confidence,
                self.low_confidence_ratio,
                self.id_number_format_valid,
                self.field_extraction_completeness,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "ocr_mean_confidence",
            "ocr_min_confidence",
            "ocr_std_confidence",
            "ocr_low_confidence_ratio",
            "ocr_id_number_format_valid",
            "ocr_field_extraction_completeness",
        ]


LOW_CONFIDENCE_THRESHOLD = 0.5


def extract_ocr_derived_features(
    ocr_results: list[OcrResult], parsed_fields: ParsedDocumentFields
) -> OcrDerivedFeatures:
    """
    Compute numeric OCR-confidence and structural-validity features for
    use in the forgery-detection feature fusion vector.
    """
    confidences = np.array([r.confidence for r in ocr_results], dtype=np.float32)

    if confidences.size == 0:
        raise ValueError(
            "Cannot compute OCR-derived features from an empty OCR result list."
        )

    mean_confidence = float(np.mean(confidences))
    min_confidence = float(np.min(confidences))
    std_confidence = float(np.std(confidences))
    low_confidence_ratio = float(
        np.sum(confidences < LOW_CONFIDENCE_THRESHOLD) / confidences.size
    )

    if parsed_fields.document_type == "AADHAAR":
        id_valid = is_valid_aadhaar_format(parsed_fields.aadhaar_number or "")
        required_fields = [
            parsed_fields.name,
            parsed_fields.dob,
            parsed_fields.gender,
            parsed_fields.aadhaar_number,
        ]
    elif parsed_fields.document_type == "PAN":
        id_valid = is_valid_pan_format(parsed_fields.pan_number or "")
        required_fields = [
            parsed_fields.name,
            parsed_fields.dob,
            parsed_fields.pan_number,
        ]
    else:
        id_valid = False
        required_fields = [parsed_fields.name, parsed_fields.dob]

    completeness = sum(1 for f in required_fields if f) / len(required_fields)

    return OcrDerivedFeatures(
        mean_confidence=mean_confidence,
        min_confidence=min_confidence,
        std_confidence=std_confidence,
        low_confidence_ratio=low_confidence_ratio,
        id_number_format_valid=1.0 if id_valid else 0.0,
        field_extraction_completeness=float(completeness),
    )
