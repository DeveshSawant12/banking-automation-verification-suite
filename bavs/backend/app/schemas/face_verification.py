"""
Pydantic schemas for Face Verification (Module 4).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FaceVerificationResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kyc_case_id: uuid.UUID
    id_document_id: uuid.UUID
    selfie_document_id: uuid.UUID
    cosine_similarity: float
    match_percentage: float
    is_match: bool
    detector_backend: str
    model_name: str
    created_at: datetime


class FaceVerificationCaseSummary(BaseModel):
    """
    Aggregate view across all ID-photo comparisons for a single KYC case
    (e.g. Aadhaar-vs-selfie AND PAN-vs-selfie if both documents were
    uploaded). overall_is_match is True only if EVERY available comparison
    matched -- a single mismatch against either ID photo is treated as a
    verification failure, since a real customer's face should match all
    of their own ID photos.
    """

    kyc_case_id: uuid.UUID
    comparisons: list[FaceVerificationResultResponse]
    overall_is_match: bool
    lowest_match_percentage: float
