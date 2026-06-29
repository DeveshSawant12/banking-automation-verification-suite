"""
Pydantic schemas for Document and OCR-related API request/response models.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models.document import DocumentType


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kyc_case_id: uuid.UUID
    document_type: DocumentType
    original_filename: str
    content_hash: str
    uploaded_at: datetime


class OcrFieldConfidences(BaseModel):
    name: float | None = None


class OcrExtractionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    extracted_name: str | None
    extracted_dob: str | None
    extracted_gender: str | None
    extracted_address: str | None
    extracted_aadhaar_no: str | None
    extracted_pan_no: str | None
    created_at: datetime


class OcrExtractionResult(BaseModel):
    """
    Internal result object returned by ocr_service.extract_document_fields
    before persistence — includes warnings that are not stored as DB
    columns but ARE surfaced to the API caller for transparency.
    """

    document_type: str
    extracted_name: str | None = None
    extracted_dob: str | None = None
    extracted_gender: str | None = None
    extracted_address: str | None = None
    extracted_aadhaar_no: str | None = None
    extracted_pan_no: str | None = None
    field_confidences: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    raw_ocr_json: dict
