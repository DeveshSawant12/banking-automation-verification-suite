"""
Cross Document Verification Service — Module 5 orchestration layer.

Pipeline (per locked spec):
    Fetch OcrExtraction rows for the AADHAAR and PAN documents of a KYC
    case -> compare Name (RapidFuzz token_sort_ratio) and DOB (exact
    match) -> combine into a weighted overall_match_score (Name 70% /
    DOB 30%, per locked decision) -> persist CrossDocumentResult.

Gender is reported but NOT included in overall_match_score, since PAN
cards carry no gender field (see cross_document_result.py docstring for
the schema rationale). gender_match is persisted as None
(not-applicable) whenever the PAN document's extracted_gender is absent,
which is the expected/normal case for every real PAN card.

No silent failure paths: if either the AADHAAR or PAN OcrExtraction is
missing (i.e. OCR wasn't run yet, or the document wasn't uploaded), this
raises CrossDocumentServiceError rather than fabricating a comparison
against absent data.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.orm import Session

from app.db.models.document import Document, DocumentType
from app.db.models.cross_document_result import CrossDocumentResult
from app.db.models.ocr_extraction import OcrExtraction
from app.services.audit_service import AuditEventType, write_audit_log
from app.utils.fuzzy_match import dob_exact_match, gender_exact_match, name_similarity_score

logger = logging.getLogger(__name__)

NAME_WEIGHT = 0.70
DOB_WEIGHT = 0.30


class CrossDocumentServiceError(Exception):
    """Raised when cross-document verification cannot be completed."""


def _get_ocr_extraction_for_document_type(
    db: Session, kyc_case_id: uuid.UUID, document_type: DocumentType
) -> OcrExtraction | None:
    """
    Fetch the OcrExtraction row for the given KYC case's document of the
    specified type. Returns None if no such document exists, or if the
    document exists but OCR has not yet been run on it.
    """
    document = (
        db.query(Document)
        .filter(
            Document.kyc_case_id == kyc_case_id,
            Document.document_type == document_type,
        )
        .one_or_none()
    )
    if document is None or document.ocr_extraction is None:
        return None
    return document.ocr_extraction


def compare_aadhaar_pan_fields(
    aadhaar_extraction: OcrExtraction, pan_extraction: OcrExtraction
) -> dict:
    """
    Compare Name and DOB between an Aadhaar and PAN OcrExtraction, and
    report Gender (which is always not-applicable for this specific
    comparison, since PAN has no gender field — included for schema
    completeness and to make the N/A status explicit rather than absent).

    Returns:
        dict with keys: name_match_score (float), dob_match (bool),
        gender_match (bool | None), overall_match_score (float)
    """
    name_score = name_similarity_score(
        aadhaar_extraction.extracted_name, pan_extraction.extracted_name
    )
    dob_matches = dob_exact_match(
        aadhaar_extraction.extracted_dob, pan_extraction.extracted_dob
    )

    # PAN extraction never populates extracted_gender (Module 1 field_parser
    # only attempts gender extraction for AADHAAR document_type). This is
    # always None for the current Aadhaar-vs-PAN comparison scope, but the
    # check is written explicitly (not just hardcoded None) so this function
    # behaves correctly if PAN extraction logic ever changes upstream.
    gender_value = pan_extraction.extracted_gender
    if gender_value is None:
        gender_matches = None
    else:
        gender_matches = gender_exact_match(
            aadhaar_extraction.extracted_gender, gender_value
        )

    dob_score = 100.0 if dob_matches else 0.0
    overall_score = (name_score * NAME_WEIGHT) + (dob_score * DOB_WEIGHT)

    return {
        "name_match_score": name_score,
        "dob_match": dob_matches,
        "gender_match": gender_matches,
        "overall_match_score": round(overall_score, 2),
    }


def persist_cross_document_result(
    db: Session, kyc_case_id: uuid.UUID, comparison: dict
) -> CrossDocumentResult:
    """
    Persist a cross-document comparison result. Raises if a result
    already exists for this case (cross-document verification should
    run once per case, not be silently duplicated).
    """
    existing = (
        db.query(CrossDocumentResult)
        .filter(CrossDocumentResult.kyc_case_id == kyc_case_id)
        .one_or_none()
    )
    if existing is not None:
        raise CrossDocumentServiceError(
            f"Cross-document result already exists for case {kyc_case_id}."
        )

    result = CrossDocumentResult(
        kyc_case_id=kyc_case_id,
        name_match_score=comparison["name_match_score"],
        dob_match=comparison["dob_match"],
        gender_match=comparison["gender_match"],
        overall_match_score=comparison["overall_match_score"],
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def run_cross_document_verification(
    db: Session, kyc_case_id: uuid.UUID
) -> CrossDocumentResult:
    """
    Full Module 5 entry point: fetches both documents' OCR extractions
    for the given case, compares them, and persists the result.

    Raises:
        CrossDocumentServiceError: if either the AADHAAR or PAN document's
            OCR extraction is missing. Cross-document verification cannot
            run on incomplete data — the orchestrator must ensure Module 1
            (OCR) has completed for both documents before calling this.
    """
    aadhaar_extraction = _get_ocr_extraction_for_document_type(
        db, kyc_case_id, DocumentType.AADHAAR
    )
    if aadhaar_extraction is None:
        raise CrossDocumentServiceError(
            f"No completed Aadhaar OCR extraction found for case {kyc_case_id}. "
            f"Cross-document verification requires OCR to have run on both "
            f"the Aadhaar and PAN documents first."
        )

    pan_extraction = _get_ocr_extraction_for_document_type(
        db, kyc_case_id, DocumentType.PAN
    )
    if pan_extraction is None:
        raise CrossDocumentServiceError(
            f"No completed PAN OCR extraction found for case {kyc_case_id}. "
            f"Cross-document verification requires OCR to have run on both "
            f"the Aadhaar and PAN documents first."
        )

    comparison = compare_aadhaar_pan_fields(aadhaar_extraction, pan_extraction)

    logger.info(
        "Cross-document verification: case=%s name_score=%.2f dob_match=%s "
        "gender_match=%s overall=%.2f",
        kyc_case_id,
        comparison["name_match_score"],
        comparison["dob_match"],
        comparison["gender_match"],
        comparison["overall_match_score"],
    )

    result = persist_cross_document_result(db, kyc_case_id, comparison)

    # Module 9 wiring: write an audit entry at this natural completion point.
    write_audit_log(
        db,
        AuditEventType.CROSS_DOCUMENT_VERIFICATION_COMPLETED,
        kyc_case_id=kyc_case_id,
        verification_status=f"overall_score={comparison['overall_match_score']:.2f}",
        metadata_json={
            "name_match_score": comparison["name_match_score"],
            "dob_match": comparison["dob_match"],
            "gender_match": comparison["gender_match"],
            "overall_match_score": comparison["overall_match_score"],
        },
    )

    return result
