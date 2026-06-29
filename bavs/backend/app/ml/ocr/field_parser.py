"""
Structured field parser for OCR output.

Aadhaar and PAN cards have different, well-documented layouts:

AADHAAR (UIDAI format):
    - Name printed in English (and often regional language above it)
    - DOB or "Year of Birth" labeled line
    - Gender (MALE/FEMALE/TRANSGENDER)
    - 12-digit Aadhaar number, grouped in 4s, usually near bottom
    - Address block (only on the address-bearing side of the card),
      typically prefixed with "Address:" and ending in a 6-digit PIN code

PAN (Income Tax Department format):
    - "Permanent Account Number" / "Income Tax Department" header text
    - Name (printed in capitals)
    - Father's Name (printed below Name — used to avoid misclassifying it
      as the holder's name)
    - DOB
    - 10-character PAN number
    - No address field on a PAN card (this is a real, documented fact —
      PAN cards do not print address; only Aadhaar does)

This module determines document type from text content (not filename,
since filenames are user-supplied and untrustworthy) and extracts the
fields enumerated in Module 1 of the spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ml.ocr.easyocr_engine import OcrResult, get_full_text
from app.utils.regex_patterns import (
    extract_aadhaar_number,
    extract_dob,
    extract_gender,
    extract_pan_number,
    is_valid_aadhaar_format,
    is_valid_pan_format,
)

PAN_HEADER_KEYWORDS = ("INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "GOVT. OF INDIA")
AADHAAR_HEADER_KEYWORDS = ("UNIQUE IDENTIFICATION AUTHORITY", "GOVERNMENT OF INDIA", "AADHAAR")
ADDRESS_LABEL_KEYWORDS = ("ADDRESS", "पता")
FATHER_NAME_LABEL_KEYWORDS = ("FATHER", "FATHER'S NAME", "S/O")


@dataclass
class ParsedDocumentFields:
    document_type: str  # "AADHAAR" | "PAN" | "UNKNOWN"
    name: str | None = None
    dob: str | None = None
    gender: str | None = None
    address: str | None = None
    aadhaar_number: str | None = None
    pan_number: str | None = None
    field_confidences: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def detect_document_type(full_text: str) -> str:
    """
    Classify the document as AADHAAR, PAN, or UNKNOWN based on header
    keywords and the presence of a structurally valid number for each type.
    """
    upper_text = full_text.upper()

    has_pan_keywords = any(kw in upper_text for kw in PAN_HEADER_KEYWORDS)
    has_aadhaar_keywords = any(kw in upper_text for kw in AADHAAR_HEADER_KEYWORDS)

    pan_number = extract_pan_number(full_text)
    aadhaar_number = extract_aadhaar_number(full_text)

    has_valid_pan = pan_number is not None and is_valid_pan_format(pan_number)
    has_valid_aadhaar = (
        aadhaar_number is not None and is_valid_aadhaar_format(aadhaar_number)
    )

    if has_pan_keywords or (has_valid_pan and not has_valid_aadhaar):
        return "PAN"
    if has_aadhaar_keywords or (has_valid_aadhaar and not has_valid_pan):
        return "AADHAAR"

    return "UNKNOWN"


def _extract_name_aadhaar(ocr_results: list[OcrResult]) -> tuple[str | None, float]:
    """
    Heuristic: on Aadhaar cards, the holder's name is typically the first
    all-uppercase English text line of reasonable length (2+ words or
    single word >= 3 chars) that is not a header keyword and does not
    contain digits. Returns (name, confidence_of_that_ocr_line).
    """
    for result in ocr_results:
        text = result.text.strip()
        upper = text.upper()

        if any(kw in upper for kw in AADHAAR_HEADER_KEYWORDS):
            continue
        if any(char.isdigit() for char in text):
            continue
        if len(text) < 3:
            continue
        if text != upper:
            # Skip non-uppercase lines (often the regional-language name,
            # which we are not attempting to parse here).
            continue

        return text, result.confidence

    return None, 0.0


def _extract_name_pan(ocr_results: list[OcrResult]) -> tuple[str | None, float]:
    for i, result in enumerate(ocr_results):
        text = result.text.strip().upper()

        if text == "NAME":
            if i + 1 < len(ocr_results):
                candidate = ocr_results[i + 1]
                candidate_text = candidate.text.strip()

                if re.fullmatch(r"[A-Z ]{5,}", candidate_text):
                    return candidate_text, candidate.confidence

    return None, 0.0


def _extract_address(ocr_results: list[OcrResult], full_text: str) -> str | None:
    """
    Extract the address block on Aadhaar cards: text following an
    "Address:" label line, concatenated until a 6-digit PIN code is
    encountered (which marks the end of an Indian postal address).
    """
    lines = [r.text.strip() for r in ocr_results]
    address_start_idx = None

    for idx, line in enumerate(lines):
        if any(kw in line.upper() for kw in ADDRESS_LABEL_KEYWORDS):
            address_start_idx = idx
            break

    if address_start_idx is None:
        return None

    address_parts = []
    for line in lines[address_start_idx:]:
        cleaned = line
        for kw in ADDRESS_LABEL_KEYWORDS:
            # Case-insensitive removal of the label word itself (e.g. "Address:"
            # printed on the card, vs. ADDRESS_LABEL_KEYWORDS stored uppercase).
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            cleaned = pattern.sub("", cleaned)
        cleaned = cleaned.replace(":", "").strip()
        if cleaned:
            address_parts.append(cleaned)
        if any(char.isdigit() for char in cleaned) and len(
            "".join(filter(str.isdigit, cleaned))
        ) == 6:
            break

    if not address_parts:
        return None
    return ", ".join(address_parts)


def parse_fields(ocr_results: list[OcrResult]) -> ParsedDocumentFields:
    """
    Main entry point: takes raw EasyOCR results and produces structured,
    validated fields per Module 1 requirements.
    """
    full_text = get_full_text(ocr_results)
    doc_type = detect_document_type(full_text)

    parsed = ParsedDocumentFields(document_type=doc_type)

    parsed.dob = extract_dob(full_text)
    parsed.gender = extract_gender(full_text)

    if doc_type == "AADHAAR":
        aadhaar_number = extract_aadhaar_number(full_text)
        if aadhaar_number and is_valid_aadhaar_format(aadhaar_number):
            parsed.aadhaar_number = aadhaar_number
        else:
            parsed.warnings.append(
                "Could not extract a structurally valid Aadhaar number."
            )

        name, name_conf = _extract_name_aadhaar(ocr_results)
        parsed.name = name
        if name_conf:
            parsed.field_confidences["name"] = round(name_conf, 4)

        parsed.address = _extract_address(ocr_results, full_text)
        if parsed.address is None:
            parsed.warnings.append(
                "No address block detected (expected on Aadhaar front/back)."
            )

    elif doc_type == "PAN":
        pan_number = extract_pan_number(full_text)
        if pan_number and is_valid_pan_format(pan_number):
            parsed.pan_number = pan_number
        else:
            parsed.warnings.append(
                "Could not extract a structurally valid PAN number."
            )

        name, name_conf = _extract_name_pan(ocr_results)
        parsed.name = name
        if name_conf:
            parsed.field_confidences["name"] = round(name_conf, 4)

    else:
        parsed.warnings.append(
            "Document type could not be determined from OCR text. "
            "Neither valid Aadhaar nor PAN markers were found."
        )

    if parsed.dob is None:
        parsed.warnings.append("Could not extract DOB.")
    if parsed.gender is None and doc_type == "AADHAAR":
        parsed.warnings.append("Could not extract gender.")
    if parsed.name is None:
        parsed.warnings.append("Could not extract name.")

    return parsed
