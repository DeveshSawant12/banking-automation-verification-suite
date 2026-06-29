"""
Document model — metadata for uploaded Aadhaar/PAN/Selfie files.

Raw file bytes are NEVER stored in Postgres; only the R2 object key is
stored (r2_object_key), per the storage architecture decision (Cloudflare
R2). content_hash (SHA-256) is stored for tamper-evidence and to detect
duplicate uploads.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DocumentType(str, enum.Enum):
    AADHAAR = "AADHAAR"
    PAN = "PAN"
    SELFIE = "SELFIE"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kyc_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("kyc_cases.id", ondelete="CASCADE"), nullable=False
    )
    document_type: Mapped[DocumentType] = mapped_column(
        Enum(DocumentType, name="document_type"), nullable=False
    )
    r2_object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    kyc_case: Mapped["KycCase"] = relationship("KycCase", back_populates="documents")
    ocr_extraction: Mapped["OcrExtraction | None"] = relationship(
        "OcrExtraction",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan",
    )
    tampering_result: Mapped["TamperingResult | None"] = relationship(
        "TamperingResult",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan",
    )
