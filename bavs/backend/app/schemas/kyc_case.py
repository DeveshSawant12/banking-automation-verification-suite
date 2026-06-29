"""KYC Case Pydantic schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models.kyc_case import KycStatus


class KycCaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    customer_id: uuid.UUID
    status: KycStatus
    assigned_staff_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime
    decided_at: datetime | None


class KycStatusResponse(BaseModel):
    id: uuid.UUID
    status: KycStatus


class CaseDecisionRequest(BaseModel):
    decision: str  # "APPROVED" | "REJECTED"
