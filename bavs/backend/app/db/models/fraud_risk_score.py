"""
FraudRiskScore model — one row per kyc_case_id, storing the aggregated
fraud risk score computed by Module 7's rules engine.

Matches the original locked schema (total_score, risk_level,
score_breakdown JSONB) exactly. score_breakdown stores a key-per-rule
mapping (e.g. {"tampered_aadhaar": 50, "name_mismatch": 20}) so the
Admin Dashboard (Module 11) and Explainable AI layer (Module 8) can
display exactly which rules fired without re-deriving them from raw
verification tables.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class RiskLevel(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FraudRiskScore(Base):
    __tablename__ = "fraud_risk_scores"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kyc_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("kyc_cases.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    total_score: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_level: Mapped[RiskLevel] = mapped_column(
        Enum(RiskLevel, name="risk_level"), nullable=False
    )
    score_breakdown: Mapped[dict] = mapped_column(JSONB, nullable=False)
    requires_manual_review: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    kyc_case: Mapped["KycCase"] = relationship("KycCase")
