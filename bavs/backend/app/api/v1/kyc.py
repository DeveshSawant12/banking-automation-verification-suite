"""
KYC router — case lifecycle and document upload.

POST  /kyc/cases                           create a new case
POST  /kyc/cases/{case_id}/documents       upload Aadhaar / PAN / Selfie
POST  /kyc/cases/{case_id}/run-pipeline    trigger the verification pipeline
GET   /kyc/cases/{case_id}/status          poll pipeline status
GET   /kyc/cases/{case_id}                 full case detail
GET   /kyc/cases                           list cases (staff/admin)
PATCH /kyc/cases/{case_id}/decision        manual approve/reject (staff/admin)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.models.document import Document, DocumentType
from app.db.models.kyc_case import KycCase, KycStatus
from app.db.models.user import User, UserRole
from app.db.session import get_db
from app.dependencies import get_current_user, require_roles
from app.schemas.document import DocumentUploadResponse
from app.schemas.kyc_case import CaseDecisionRequest, KycCaseResponse, KycStatusResponse
from app.services.audit_service import AuditEventType, write_audit_log
from app.services.storage_service import upload_bytes
from app.utils.image_utils import compute_sha256

router = APIRouter(prefix="/kyc", tags=["kyc"])


@router.post("/cases", response_model=KycCaseResponse, status_code=status.HTTP_201_CREATED)
def create_case(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = KycCase(customer_id=current_user.id, status=KycStatus.PENDING)
    db.add(case)
    db.commit()
    db.refresh(case)
    write_audit_log(
        db, AuditEventType.KYC_CASE_CREATED,
        kyc_case_id=case.id, user_id=current_user.id,
    )
    return case


@router.post("/cases/{case_id}/documents", response_model=DocumentUploadResponse)
async def upload_document(
    case_id: uuid.UUID,
    document_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = db.get(KycCase, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    if current_user.role == UserRole.CUSTOMER and case.customer_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your case.")

    try:
        doc_type = DocumentType(document_type.upper())
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid document_type. Must be one of: {[d.value for d in DocumentType]}",
        )

    file_bytes = await file.read()
    content_hash = compute_sha256(file_bytes)

    r2_key = upload_bytes(
        file_bytes,
        key_prefix=f"documents/{case_id}/{doc_type.value.lower()}",
        content_type=file.content_type or "application/octet-stream",
    )

    doc = Document(
        kyc_case_id=case_id,
        document_type=doc_type,
        r2_object_key=r2_key,
        original_filename=file.filename or "upload",
        content_hash=content_hash,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    write_audit_log(
        db, AuditEventType.DOCUMENT_UPLOADED,
        kyc_case_id=case_id, user_id=current_user.id,
        metadata_json={"document_type": doc_type.value, "document_id": str(doc.id)},
    )
    return doc


@router.post("/cases/{case_id}/run-pipeline", status_code=status.HTTP_202_ACCEPTED)
def run_pipeline(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Trigger the full KYC verification pipeline as an async Celery task.
    Returns 202 Accepted immediately — poll /status to track progress.
    """
    case = db.get(KycCase, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    if current_user.role == UserRole.CUSTOMER and case.customer_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your case.")
    if case.status not in (KycStatus.PENDING, KycStatus.REVIEW_REQUIRED):
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline cannot be re-run from status '{case.status.value}'.",
        )

    case.status = KycStatus.PROCESSING
    db.commit()

    # Import here to avoid circular imports at module load time.
    from app.tasks.celery_worker import run_kyc_pipeline_task
    run_kyc_pipeline_task.delay(str(case_id))

    return {"case_id": str(case_id), "status": KycStatus.PROCESSING.value}


@router.get("/cases/{case_id}/status", response_model=KycStatusResponse)
def get_case_status(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = db.get(KycCase, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    if current_user.role == UserRole.CUSTOMER and case.customer_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your case.")
    return KycStatusResponse(id=case.id, status=case.status)


@router.get("/cases/{case_id}", response_model=KycCaseResponse)
def get_case(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    case = db.get(KycCase, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    if current_user.role == UserRole.CUSTOMER and case.customer_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your case.")
    return case


@router.get("/cases", response_model=list[KycCaseResponse])
def list_cases(
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.BANK_STAFF)),
):
    query = db.query(KycCase)
    if status_filter:
        try:
            query = query.filter(KycCase.status == KycStatus(status_filter.upper()))
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid status: {status_filter}")
    return query.order_by(KycCase.created_at.desc()).offset(offset).limit(limit).all()


@router.patch("/cases/{case_id}/decision", response_model=KycCaseResponse)
def make_decision(
    case_id: uuid.UUID,
    body: CaseDecisionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.ADMIN, UserRole.BANK_STAFF)),
):
    case = db.get(KycCase, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")

    allowed_decisions = {
        "APPROVED": KycStatus.APPROVED,
        "REJECTED": KycStatus.REJECTED,
    }
    if body.decision.upper() not in allowed_decisions:
        raise HTTPException(
            status_code=422,
            detail=f"Decision must be one of: {list(allowed_decisions.keys())}",
        )

    case.status = allowed_decisions[body.decision.upper()]
    case.decided_at = datetime.now(timezone.utc)
    case.assigned_staff_id = current_user.id
    db.commit()
    db.refresh(case)

    write_audit_log(
        db, AuditEventType.KYC_CASE_DECISION_MADE,
        kyc_case_id=case_id, user_id=current_user.id,
        decision=body.decision.upper(),
    )
    return case
