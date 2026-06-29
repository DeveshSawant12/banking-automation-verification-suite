"""
RapidFuzz-based fuzzy matching utilities for cross-document field
comparison (Module 5).

Name comparison uses token_sort_ratio (per locked decision) — this
tokenizes both strings, sorts tokens alphabetically, then computes a
Levenshtein-based ratio, so word-order differences (e.g. "RAHUL SHARMA"
vs "SHARMA RAHUL", or extra middle names appearing in different
positions across Aadhaar vs PAN OCR output) don't unfairly penalize a
genuine match.

DOB comparison is NOT fuzzy-matched — dates are compared via exact
string equality after normalization to DD/MM/YYYY (the format already
guaranteed by app.utils.regex_patterns.extract_dob in Module 1). Fuzzy-
matching a date string would be meaningless/dangerous (e.g. "15/08/1995"
vs "16/08/1995" might score deceptively high on character similarity
despite being a completely different date) — exact match is the only
sound comparison for DOB.
"""

from __future__ import annotations

from rapidfuzz import fuzz


def name_similarity_score(name_a: str | None, name_b: str | None) -> float:
    """
    Compute name similarity using RapidFuzz token_sort_ratio.

    Returns:
        float in [0, 100]. Returns 0.0 if either name is None/empty
        (cannot meaningfully compare a missing name — treated as a full
        mismatch, never silently skipped).
    """
    if not name_a or not name_b:
        return 0.0

    normalized_a = name_a.strip().upper()
    normalized_b = name_b.strip().upper()

    return float(fuzz.token_sort_ratio(normalized_a, normalized_b))


def dob_exact_match(dob_a: str | None, dob_b: str | None) -> bool:
    """
    Compare two DOB strings for exact equality after normalization.

    Both inputs are expected in DD/MM/YYYY format (guaranteed by
    app.utils.regex_patterns.extract_dob), but this function defensively
    re-normalizes separators (- vs /) in case a caller passes a raw OCR
    string that hasn't gone through extract_dob.

    Returns:
        False if either DOB is None/empty (cannot confirm a match against
        missing data — never silently treated as matching).
    """
    if not dob_a or not dob_b:
        return False

    normalized_a = dob_a.strip().replace("-", "/")
    normalized_b = dob_b.strip().replace("-", "/")

    return normalized_a == normalized_b


def gender_exact_match(gender_a: str | None, gender_b: str | None) -> bool:
    """
    Compare two gender strings for exact equality (case-insensitive).
    Used only when both documents actually carry a gender field (e.g. a
    future document type comparison) — PAN cards do not print gender, so
    this is not invoked for Aadhaar-vs-PAN comparison in Module 5's
    current scope.

    Returns:
        False if either value is None/empty.
    """
    if not gender_a or not gender_b:
        return False

    return gender_a.strip().upper() == gender_b.strip().upper()
