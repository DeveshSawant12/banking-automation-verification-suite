"""
Pydantic schemas for Module 9 (Audit Logs).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kyc_case_id: uuid.UUID | None
    user_id: uuid.UUID | None
    event_type: str
    verification_status: str | None
    risk_score: int | None
    face_match_pct: float | None
    decision: str | None
    metadata_json: dict | None
    timestamp: datetime
