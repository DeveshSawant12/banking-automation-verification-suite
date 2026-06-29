"""
OcrExtraction model — one row per document, storing the structured fields
parsed by app/ml/ocr/field_parser.py plus the full raw EasyOCR JSON output
(for debugging, re-parsing without re-running OCR, and audit purposes).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class OcrExtraction(Base):
    __tablename__ = "ocr_extractions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    extracted_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    extracted_dob: Mapped[str | None] = mapped_column(String(20), nullable=True)
    extracted_gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    extracted_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_aadhaar_no: Mapped[str | None] = mapped_column(String(20), nullable=True)
    extracted_pan_no: Mapped[str | None] = mapped_column(String(20), nullable=True)
    raw_ocr_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document: Mapped["Document"] = relationship(
        "Document", back_populates="ocr_extraction"
    )
