"""
Audit Logging Service — Module 9.

Standalone, callable audit-entry writer. This module does NOT call any
other service; it is called BY other services (OCR, tampering, face
verification, cross-document, liveness, fraud risk, explainability) at
their natural completion points, per the locked decision to wire actual
call-sites into Modules 1-8 now rather than defer to an orchestrator that
doesn't exist yet.

EVENT TYPE TAXONOMY: rather than let call-sites pass arbitrary free-text
event_type strings (which would make the audit log un-queryable and
inconsistent across modules), this module defines a closed enum of real
event types corresponding to what each module actually does. This is the
single source of truth for what's loggable — a new event type requires a
deliberate addition here, not an ad-hoc string at a call site.

INSERT-ONLY BY DESIGN: this module exposes ONLY a create function. There
is no update_audit_log or delete_audit_log function, by design — pairing
the application-layer omission with the database-level REVOKE enforced
in alembic/versions/f1a9c3d7e2b4_audit_log_insert_only.py (see that
migration's docstring for why both layers matter).

FAILURE HANDLING: audit logging failures must NEVER abort the underlying
verification operation that triggered them — a banking KYC pipeline must
not fail a legitimate customer's onboarding because, say, a transient DB
write to the audit table failed, while the actual OCR/tampering/etc.
result was computed successfully. write_audit_log therefore catches and
logs (rather than propagates) database errors. This is a deliberate
asymmetry from this project's usual "raise, don't swallow" pattern: it
applies specifically to audit logging because the consequence of a
swallowed audit-write failure (a gap in the audit trail, logged loudly to
the application's own logs) is categorically less severe than the
consequence of blocking a real verification result on a logging
side-effect.
"""

from __future__ import annotations

import enum
import logging
import uuid

from sqlalchemy.orm import Session

from app.db.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


class AuditEventType(str, enum.Enum):
    # Module 1
    OCR_EXTRACTION_COMPLETED = "OCR_EXTRACTION_COMPLETED"
    OCR_EXTRACTION_FAILED = "OCR_EXTRACTION_FAILED"
    # Module 2 / 3
    AADHAAR_TAMPERING_CHECK_COMPLETED = "AADHAAR_TAMPERING_CHECK_COMPLETED"
    PAN_TAMPERING_CHECK_COMPLETED = "PAN_TAMPERING_CHECK_COMPLETED"
    # Module 4
    FACE_VERIFICATION_COMPLETED = "FACE_VERIFICATION_COMPLETED"
    # Module 5
    CROSS_DOCUMENT_VERIFICATION_COMPLETED = "CROSS_DOCUMENT_VERIFICATION_COMPLETED"
    # Module 6
    LIVENESS_CHECK_COMPLETED = "LIVENESS_CHECK_COMPLETED"
    # Module 7
    FRAUD_RISK_SCORED = "FRAUD_RISK_SCORED"
    # Module 8
    EXPLANATION_GENERATED = "EXPLANATION_GENERATED"
    # Case-level lifecycle (for Module 11 / future orchestrator use)
    KYC_CASE_CREATED = "KYC_CASE_CREATED"
    KYC_CASE_DECISION_MADE = "KYC_CASE_DECISION_MADE"
    DOCUMENT_UPLOADED = "DOCUMENT_UPLOADED"


def write_audit_log(
    db: Session,
    event_type: AuditEventType,
    kyc_case_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
    verification_status: str | None = None,
    risk_score: int | None = None,
    face_match_pct: float | None = None,
    decision: str | None = None,
    metadata_json: dict | None = None,
) -> AuditLog | None:
    """
    Write a single audit log entry. Returns the persisted AuditLog row,
    or None if the write failed (failure is logged, never raised — see
    module docstring on why audit-write failures must not block the
    underlying operation).

    CALLER REQUIREMENT (found during testing, not theoretical): pass
    kyc_case_id and user_id as plain uuid.UUID values, captured from an
    ORM object's .id BEFORE that object's owning session may have been
    closed or its attributes expired — e.g. `case_id = case.id` captured
    immediately after a commit, then passed here, rather than passing
    `case.id` directly at the call site if there's any chance the
    session has since been closed/expired. SQLAlchemy's lazy attribute
    loading means `case.id` can raise DetachedInstanceError at the
    EXPRESSION-EVALUATION point (before this function's own try/except
    even begins executing) if `case`'s session is closed and its `id`
    attribute has been expired since it was last accessed. This
    function's try/except can only contain failures that occur INSIDE
    its own body — it cannot contain a failure in evaluating its own
    arguments. This is standard Python/SQLAlchemy behavior, not a defect
    in this function, but it is documented here explicitly because it is
    exactly the kind of failure mode that would otherwise surface as a
    confusing crash deep in a Celery task calling this function with a
    stale ORM reference.

    This function performs its OWN commit (rather than relying on the
    caller's transaction), so an audit-write failure cannot roll back or
    interfere with the caller's own already-committed verification
    result. This is intentional: by the time write_audit_log is called,
    the actual verification work (OCR/tampering/etc.) has already been
    persisted by the calling service's own commit.
    """
    try:
        entry = AuditLog(
            kyc_case_id=kyc_case_id,
            user_id=user_id,
            event_type=event_type.value,
            verification_status=verification_status,
            risk_score=risk_score,
            face_match_pct=face_match_pct,
            decision=decision,
            metadata_json=metadata_json,
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        return entry
    except Exception as exc:
        # Broad except is deliberate here: ANY failure writing an audit
        # entry (DB connectivity, constraint violation, etc.) must be
        # contained to logging, never propagated to block the caller's
        # already-successful verification result. db.rollback() ensures
        # a failed audit write doesn't leave the session in a broken
        # state for whatever the caller does next.
        db.rollback()
        logger.error(
            "Failed to write audit log entry (event_type=%s, "
            "kyc_case_id=%s): %s. This indicates a gap in the audit "
            "trail that should be investigated, but the underlying "
            "verification operation was NOT affected.",
            event_type.value,
            kyc_case_id,
            exc,
        )
        return None


def get_audit_logs_for_case(db: Session, kyc_case_id: uuid.UUID) -> list[AuditLog]:
    """
    Retrieve all audit log entries for a given KYC case, ordered
    chronologically. Read-only — consistent with this module exposing no
    update/delete functions.
    """
    return (
        db.query(AuditLog)
        .filter(AuditLog.kyc_case_id == kyc_case_id)
        .order_by(AuditLog.timestamp.asc())
        .all()
    )


def get_audit_logs_paginated(
    db: Session, limit: int = 50, offset: int = 0
) -> list[AuditLog]:
    """
    Retrieve audit log entries across all cases, most recent first, for
    the Admin Dashboard's audit log view (Module 11). Read-only.
    """
    return (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
