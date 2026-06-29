"""
OCR Service — orchestration layer for Module 1.

Responsible for the full pipeline from raw uploaded bytes to a persisted
OcrExtraction row:

    raw bytes -> preprocess_for_ocr -> run_ocr -> parse_fields -> save

This is intentionally synchronous/callable code (not a Celery task itself)
so it can be:
  (a) invoked directly from a FastAPI endpoint for quick single-document
      testing/debugging, and
  (b) wrapped by a Celery task in the orchestrator module (built once all
      verification modules exist) without duplicating logic.

No silent failure paths: if OCR produces no usable text, this raises
rather than persisting an empty/garbage row, since a downstream fraud
decision must never be based on silently-failed extraction.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.orm import Session

from app.db.models.document import Document
from app.db.models.ocr_extraction import OcrExtraction
from app.ml.ocr.easyocr_engine import run_ocr
from app.ml.ocr.field_parser import parse_fields
from app.schemas.document import OcrExtractionResult
from app.services.audit_service import AuditEventType, write_audit_log
from app.utils.image_utils import ImageLoadError, preprocess_for_ocr

logger = logging.getLogger(__name__)


class OcrServiceError(Exception):
    """Raised when OCR extraction cannot be completed for a document."""


def extract_document_fields(file_bytes: bytes) -> OcrExtractionResult:
    """
    Run the full OCR extraction pipeline on raw image bytes and return a
    structured result. Does NOT touch the database — pure function over
    bytes -> result, so it is independently testable.

    Raises:
        OcrServiceError: on any failure in decoding, OCR, or parsing.
    """
    try:
        preprocessed_image = preprocess_for_ocr(file_bytes)
    except ImageLoadError as exc:
        raise OcrServiceError(f"Image preprocessing failed: {exc}") from exc

    try:
        ocr_results = run_ocr(preprocessed_image)
    except RuntimeError as exc:
        raise OcrServiceError(f"OCR extraction failed: {exc}") from exc

    parsed = parse_fields(ocr_results)

    raw_ocr_json = {
        "detections": [r.to_dict() for r in ocr_results],
        "detection_count": len(ocr_results),
    }

    return OcrExtractionResult(
        document_type=parsed.document_type,
        extracted_name=parsed.name,
        extracted_dob=parsed.dob,
        extracted_gender=parsed.gender,
        extracted_address=parsed.address,
        extracted_aadhaar_no=parsed.aadhaar_number,
        extracted_pan_no=parsed.pan_number,
        field_confidences=parsed.field_confidences,
        warnings=parsed.warnings,
        raw_ocr_json=raw_ocr_json,
    )


def persist_ocr_extraction(
    db: Session, document_id: uuid.UUID, result: OcrExtractionResult
) -> OcrExtraction:
    """
    Persist an OcrExtractionResult as an OcrExtraction row linked to the
    given document_id. Raises if the document does not exist (FK
    integrity must hold — never insert orphaned extraction records).
    """
    document = db.get(Document, document_id)
    if document is None:
        raise OcrServiceError(f"Document {document_id} does not exist.")

    existing = (
        db.query(OcrExtraction)
        .filter(OcrExtraction.document_id == document_id)
        .one_or_none()
    )

    if existing:
        db.delete(existing)
        db.commit()

    extraction = OcrExtraction(
        document_id=document_id,
        extracted_name=result.extracted_name,
        extracted_dob=result.extracted_dob,
        extracted_gender=result.extracted_gender,
        extracted_address=result.extracted_address,
        extracted_aadhaar_no=result.extracted_aadhaar_no,
        extracted_pan_no=result.extracted_pan_no,
        raw_ocr_json=result.raw_ocr_json,
    )
    db.add(extraction)
    db.commit()
    db.refresh(extraction)

    if result.warnings:
        logger.warning(
            "OCR extraction for document %s completed with warnings: %s",
            document_id,
            result.warnings,
        )

    return extraction


def run_and_persist_ocr(
    db: Session, document_id: uuid.UUID, file_bytes: bytes
) -> tuple[OcrExtraction, list[str]]:
    """
    Convenience wrapper: runs extraction and persists in one call.
    Returns the saved OcrExtraction row plus any non-fatal warnings
    (e.g. "address not found") so the caller (API endpoint or Celery
    task) can decide how to surface them.
    """
    result = extract_document_fields(file_bytes)
    extraction = persist_ocr_extraction(db, document_id, result)

    # Module 9 wiring: write an audit entry at this natural completion
    # point. document.kyc_case_id is looked up since this function is
    # keyed on document_id, but audit_logs is keyed on kyc_case_id.
    document = db.get(Document, document_id)
    write_audit_log(
        db,
        AuditEventType.OCR_EXTRACTION_COMPLETED,
        kyc_case_id=document.kyc_case_id if document else None,
        verification_status=result.document_type,
        metadata_json={
            "document_id": str(document_id),
            "warnings": result.warnings,
        },
    )

    return extraction, result.warnings
