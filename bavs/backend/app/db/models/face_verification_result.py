"""
FaceVerificationResult model — one row per (id_document, selfie) pair
compared. Per the locked decision to compare the selfie against BOTH
Aadhaar and PAN photos when both are available, a single kyc_case_id can
have MULTIPLE FaceVerificationResult rows (one per id_document_id). This
is intentionally NOT constrained to one-row-per-case — the FK structure
already supports this without any schema change beyond what was
originally defined in the architecture phase.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class FaceVerificationResult(Base):
    __tablename__ = "face_verification_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kyc_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("kyc_cases.id", ondelete="CASCADE"), nullable=False
    )
    id_document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )
    selfie_document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False
    )
    cosine_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    match_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    is_match: Mapped[bool] = mapped_column(Boolean, nullable=False)
    detector_backend: Mapped[str] = mapped_column(
        String(50), nullable=False, default="retinaface"
    )
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    kyc_case: Mapped["KycCase"] = relationship("KycCase")
    id_document: Mapped["Document"] = relationship(
        "Document", foreign_keys=[id_document_id]
    )
    selfie_document: Mapped["Document"] = relationship(
        "Document", foreign_keys=[selfie_document_id]
    )
