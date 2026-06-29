"""
Regex patterns for Indian identity document field validation.

Format references (publicly documented, not invented):
- Aadhaar Number: 12 digits, conventionally displayed in 3 groups of 4
  (UIDAI format: XXXX XXXX XXXX). First digit is never 0 or 1.
- PAN Number: 10-character alphanumeric, fixed structure mandated by the
  Income Tax Department of India:
    [AAAAA][9999][A]
    - First 5 characters: uppercase letters
    - Next 4 characters: digits
    - Last character: uppercase letter (checksum-style, not numerically derived)
- DOB: Aadhaar/PAN cards print DOB as DD/MM/YYYY or DD-MM-YYYY.
- Gender: Printed as "MALE", "FEMALE", "TRANSGENDER", or single-letter
  abbreviations "M"/"F"/"T" depending on card print era.
"""

import re

# Aadhaar: optional spaces between groups, 12 digits total, leading digit 2-9
AADHAAR_PATTERN = re.compile(r"\b([2-9]{1}\d{3})\s?(\d{4})\s?(\d{4})\b")

# PAN: strict 10-character format
PAN_PATTERN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b")

# DOB: DD/MM/YYYY or DD-MM-YYYY, also matches single-digit day/month variants
DOB_PATTERN = re.compile(
    r"\b(0?[1-9]|[12][0-9]|3[01])[/\-](0?[1-9]|1[0-2])[/\-](\d{4})\b"
)

# Gender: matches full words and single-letter abbreviations as separate tokens
GENDER_PATTERN = re.compile(
    r"\b(MALE|FEMALE|TRANSGENDER|पुरुष|महिला)\b", re.IGNORECASE
)
GENDER_ABBREV_PATTERN = re.compile(r"(?<![A-Za-z])([MFT])(?![A-Za-z])")

# PIN code (used to help anchor address block extraction for Aadhaar)
PINCODE_PATTERN = re.compile(r"\b(\d{6})\b")


def extract_aadhaar_number(text: str) -> str | None:
    """Return normalized 12-digit Aadhaar number (no spaces) or None."""
    match = AADHAAR_PATTERN.search(text)
    if not match:
        return None
    return "".join(match.groups())


def extract_pan_number(text: str) -> str | None:
    text = text.upper()

    # OCR correction:
    # Convert O → 0 only inside the 4-digit section
    text = re.sub(
        r"([A-Z]{5})([0-9O]{4})([A-Z])",
        lambda m: m.group(1) + m.group(2).replace("O", "0") + m.group(3),
        text,
    )

    match = PAN_PATTERN.search(text)
    if not match:
        return None

    return match.group(1)


def extract_dob(text: str) -> str | None:
    """Return DOB string in DD/MM/YYYY format or None."""
    match = DOB_PATTERN.search(text)
    if not match:
        return None
    day, month, year = match.groups()
    return f"{int(day):02d}/{int(month):02d}/{year}"


def extract_gender(text: str) -> str | None:
    """Return normalized gender string (MALE/FEMALE/TRANSGENDER) or None."""
    match = GENDER_PATTERN.search(text)
    if match:
        value = match.group(1).upper()
        mapping = {"पुरुष": "MALE", "महिला": "FEMALE"}
        return mapping.get(value, value)

    abbrev_match = GENDER_ABBREV_PATTERN.search(text)
    if abbrev_match:
        mapping = {"M": "MALE", "F": "FEMALE", "T": "TRANSGENDER"}
        return mapping.get(abbrev_match.group(1).upper())
    return None


def is_valid_aadhaar_format(aadhaar_number: str) -> bool:
    """Strict structural validation of an already-extracted Aadhaar number."""
    if not aadhaar_number:
        return False
    digits_only = aadhaar_number.replace(" ", "")
    return bool(re.fullmatch(r"[2-9]\d{11}", digits_only))


def is_valid_pan_format(pan_number: str) -> bool:
    """Strict structural validation of an already-extracted PAN number."""
    if not pan_number:
        return False
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]{1}", pan_number.upper()))
