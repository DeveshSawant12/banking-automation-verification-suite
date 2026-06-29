"""
TamperingResult model — stores the Random Forest verdict, confidence, and
ELA score for a single document (Aadhaar or PAN). One row per document
(documents.id), reused by both Module 2 (Aadhaar) and Module 3 (PAN)
since the schema/pipeline is identical for both per the locked
architecture — only the trained model file differs.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class TamperVerdict(str, enum.Enum):
    REAL = "REAL"
    TAMPERED = "TAMPERED"
    INCONCLUSIVE = "INCONCLUSIVE"


class TamperingResult(Base):
    __tablename__ = "tampering_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    verdict: Mapped[TamperVerdict] = mapped_column(
        Enum(TamperVerdict, name="tamper_verdict"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    ela_score: Mapped[float] = mapped_column(Float, nullable=False)
    resnet_feature_vector_ref: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    gradcam_heatmap_r2_key: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship(
        "Document", back_populates="tampering_result"
    )
