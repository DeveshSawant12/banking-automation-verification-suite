"""
CrossDocumentResult model — stores the Module 5 comparison result between
Aadhaar and PAN OCR-extracted fields for a single KYC case.

SCHEMA NOTE (deviation from the original locked architecture, called out
explicitly per project rules): gender_match is BOOLEAN NULLABLE, not
BOOLEAN NOT NULL as originally specified. Reason: PAN cards do not print
a gender field (confirmed in Module 1's field_parser.py — PAN extraction
never populates extracted_gender), so a gender comparison against PAN is
structurally impossible. NULL represents "not applicable" honestly,
rather than fabricating True or False for a field that doesn't exist on
one of the two documents being compared. This is the ONLY schema
deviation in this module; name_match_score, dob_match, and
overall_match_score are unchanged from the original design.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class CrossDocumentResult(Base):
    __tablename__ = "cross_document_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kyc_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("kyc_cases.id", ondelete="CASCADE"), nullable=False
    )
    name_match_score: Mapped[float] = mapped_column(Float, nullable=False)
    dob_match: Mapped[bool] = mapped_column(Boolean, nullable=False)
    gender_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    overall_match_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    kyc_case: Mapped["KycCase"] = relationship("KycCase")
