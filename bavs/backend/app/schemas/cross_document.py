"""
Pydantic schemas for Cross Document Verification (Module 5).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CrossDocumentResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kyc_case_id: uuid.UUID
    name_match_score: float
    dob_match: bool
    gender_match: bool | None  # None = not applicable (e.g. PAN has no gender field)
    overall_match_score: float
    created_at: datetime
