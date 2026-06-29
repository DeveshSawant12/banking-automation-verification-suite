"""
Remaining routers: fraud, audit, chatbot, admin.

Fraud   GET  /fraud/{case_id}/score
Audit   GET  /audit/logs
        GET  /audit/logs/{case_id}
Chatbot POST /chatbot/sessions
        POST /chatbot/sessions/{id}/messages
        GET  /chatbot/sessions/{id}/messages
Admin   GET  /admin/stats/summary
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models.fraud_risk_score import FraudRiskScore
from app.db.models.kyc_case import KycCase, KycStatus
from app.db.models.user import User, UserRole
from app.db.session import get_db
from app.dependencies import get_current_user, require_roles
from app.schemas.audit import AuditLogResponse
from app.schemas.chat import (
    ChatAnswerResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionResponse,
)
from app.schemas.fraud import FraudRiskScoreResponse
from app.services.audit_service import (
    get_audit_logs_for_case,
    get_audit_logs_paginated,
)
from app.services.rag_service import (
    RagServiceError,
    create_chat_session,
    get_session_messages,
    send_message,
)

# ── Fraud ─────────────────────────────────────────────────────────
fraud_router = APIRouter(prefix="/fraud", tags=["fraud"])


@fraud_router.get("/{case_id}/score", response_model=FraudRiskScoreResponse)
def get_fraud_score(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = db.get(KycCase, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    if current_user.role == UserRole.CUSTOMER and case.customer_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your case.")

    score = (
        db.query(FraudRiskScore)
        .filter(FraudRiskScore.kyc_case_id == case_id)
        .one_or_none()
    )
    if not score:
        raise HTTPException(status_code=404, detail="Fraud score not yet computed for this case.")
    return score


# ── Audit ─────────────────────────────────────────────────────────
audit_router = APIRouter(prefix="/audit", tags=["audit"])


@audit_router.get("/logs", response_model=list[AuditLogResponse])
def list_audit_logs(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(UserRole.ADMIN)),
):
    return get_audit_logs_paginated(db, limit=limit, offset=offset)


@audit_router.get("/logs/{case_id}", response_model=list[AuditLogResponse])
def get_audit_logs_by_case(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.BANK_STAFF)),
):
    return get_audit_logs_for_case(db, case_id)


# ── Chatbot ───────────────────────────────────────────────────────
chatbot_router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@chatbot_router.post("/sessions", response_model=ChatSessionResponse, status_code=201)
def new_chat_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        session = create_chat_session(db, current_user.id)
    except RagServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return session


@chatbot_router.post("/sessions/{session_id}/messages", response_model=ChatAnswerResponse)
def chat(
    session_id: uuid.UUID,
    body: ChatMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        result = send_message(db, session_id, current_user.id, body.content)
    except RagServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ChatAnswerResponse(
        user_message=ChatMessageResponse.model_validate(result["user_message"]),
        assistant_message=ChatMessageResponse.model_validate(result["assistant_message"]),
        retrieved_chunks=result["retrieved_chunks"],
    )


@chatbot_router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
def get_chat_history(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        messages = get_session_messages(db, session_id, current_user.id)
    except RagServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return messages


# ── Admin stats ───────────────────────────────────────────────────
admin_router = APIRouter(prefix="/admin", tags=["admin"])


@admin_router.get("/stats/summary")
def admin_summary(
    db: Session = Depends(get_db),
    _: User = Depends(require_roles(UserRole.ADMIN, UserRole.BANK_STAFF)),
):
    total = db.query(KycCase).count()
    approved = db.query(KycCase).filter(KycCase.status == KycStatus.APPROVED).count()
    rejected = db.query(KycCase).filter(KycCase.status == KycStatus.REJECTED).count()
    review = db.query(KycCase).filter(KycCase.status == KycStatus.REVIEW_REQUIRED).count()
    processing = db.query(KycCase).filter(KycCase.status == KycStatus.PROCESSING).count()
    pending = db.query(KycCase).filter(KycCase.status == KycStatus.PENDING).count()

    from app.db.models.fraud_risk_score import RiskLevel
    high_risk = (
        db.query(FraudRiskScore)
        .filter(FraudRiskScore.risk_level == RiskLevel.HIGH)
        .count()
    )

    return {
        "total_cases": total,
        "approved": approved,
        "rejected": rejected,
        "review_required": review,
        "processing": processing,
        "pending": pending,
        "high_risk_cases": high_risk,
    }
