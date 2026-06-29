"""
AuditLog model — immutable, append-only audit trail entry.

Matches the original locked schema exactly:
    id, kyc_case_id, user_id, event_type, verification_status,
    risk_score, face_match_pct, decision, metadata_json, timestamp

INSERT-ONLY ENFORCEMENT: per locked decision, this is enforced at the
DATABASE level (not just by omitting update/delete methods from the
service layer, which would be trivially bypassed by a bug or a future
developer adding an update call). See
alembic/versions/<rev>_audit_log_insert_only.py, which REVOKEs UPDATE and
DELETE privileges on this table for the application's database role
after table creation. This is a real Postgres-level grant — it has no
effect in the SQLite proxy used for sandbox testing (SQLite has no
per-table GRANT/REVOKE system), so insert-only enforcement can only be
verified against a real Postgres instance, not in this sandbox. This
limitation is documented here explicitly rather than silently assumed
correct.

kyc_case_id and user_id are both NULLABLE despite usually being present,
because some audit-worthy events are not tied to a specific case (e.g. an
admin login, or a system-level event before any case exists) or not tied
to an authenticated user (e.g. an automated Celery task acting on behalf
of the system rather than a person). This matches the original schema,
which already specified both as plain UUID (nullable) foreign keys, not
NOT NULL.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kyc_case_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("kyc_cases.id"), nullable=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    verification_status: Mapped[str | None] = mapped_column(String(50), nullable=True)
    risk_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    face_match_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(50), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    kyc_case: Mapped["KycCase | None"] = relationship("KycCase")
    user: Mapped["User | None"] = relationship("User")
