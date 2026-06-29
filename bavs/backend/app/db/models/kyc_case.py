"""
KycCase model — one row per customer onboarding attempt.

Complete per the schema finalized in the architecture phase. Fields like
status will be transitioned by the pipeline orchestrator (Module to be
built when all verification modules exist), but the table structure
itself is final, not a placeholder.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class KycStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    APPROVED = "APPROVED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class KycCase(Base):
    __tablename__ = "kyc_cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    status: Mapped[KycStatus] = mapped_column(
        Enum(KycStatus, name="kyc_status"), nullable=False, default=KycStatus.PENDING
    )
    assigned_staff_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    customer: Mapped["User"] = relationship(
        "User", foreign_keys=[customer_id], back_populates="kyc_cases"
    )
    assigned_staff: Mapped["User | None"] = relationship(
        "User", foreign_keys=[assigned_staff_id]
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="kyc_case", cascade="all, delete-orphan"
    )
