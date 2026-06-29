"""
Celery worker — async KYC pipeline task.

The Celery app is defined here. Docker runs this file as a separate
worker process:
    celery -A app.tasks.celery_worker worker --loglevel=info

The single public task is run_kyc_pipeline_task, triggered by the
/kyc/cases/{case_id}/run-pipeline API endpoint after all three documents
(Aadhaar, PAN, Selfie) have been uploaded. It runs the full verification
pipeline in sequence and updates the case status at the end.

Pipeline sequence (matches the locked spec):
  1. OCR extraction on Aadhaar + PAN
  2. Aadhaar tampering check
  3. PAN tampering check
  4. Cross-document verification (field matching)
  5. Face verification (Aadhaar photo + selfie, PAN photo + selfie)
  6. Liveness check is NOT run here — liveness requires a video file
     uploaded via a separate endpoint (not yet implemented). Cases
     without a liveness result will score as INCONCLUSIVE on that check,
     correctly routing to REVIEW_REQUIRED rather than silently passing.
  7. Fraud risk scoring
  8. Case status update (APPROVED / REVIEW_REQUIRED based on score)
"""

from __future__ import annotations

import logging
import uuid

from celery import Celery

from app.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

celery_app = Celery(
    "banking_kyc",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,            # task only acked after completion, retried on worker crash
    worker_prefetch_multiplier=1,   # one task per worker at a time (CV tasks are heavy)
)


@celery_app.task(
    name="app.tasks.celery_worker.run_kyc_pipeline_task",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def run_kyc_pipeline_task(self, case_id: str) -> dict:
    """
    Full KYC verification pipeline for a single case.

    All imports are local to this function to keep module-level imports
    minimal — Celery workers import this module on startup, and we want
    that to be fast without loading the entire ML stack until an actual
    task is dispatched.
    """
    from app.db.base import SessionLocal
    from app.db.models.document import Document, DocumentType
    from app.db.models.kyc_case import KycCase, KycStatus
    from app.db.models.fraud_risk_score import RiskLevel
    from app.services.ocr_service import run_and_persist_ocr, OcrServiceError
    from app.services.aadhaar_tamper_service import run_and_persist_aadhaar_tampering_check
    from app.services.pan_tamper_service import run_and_persist_pan_tampering_check
    from app.services.cross_doc_service import run_cross_document_verification
    from app.services.face_verification_service import run_face_verification_for_case
    from app.services.fraud_risk_service import run_and_persist_fraud_risk_scoring
    from app.services.storage_service import download_bytes

    case_uuid = uuid.UUID(case_id)
    db = SessionLocal()

    try:
        case = db.get(KycCase, case_uuid)
        if not case:
            logger.error("Pipeline: case %s not found.", case_id)
            return {"error": "case_not_found"}

        def _get_doc_bytes(doc_type: DocumentType) -> tuple[Document | None, bytes | None]:
            doc = (
                db.query(Document)
                .filter(
                    Document.kyc_case_id == case_uuid,
                    Document.document_type == doc_type,
                )
                .one_or_none()
            )
            if doc is None:
                return None, None
            try:
                return doc, download_bytes(doc.r2_object_key)
            except Exception as exc:
                logger.error("Pipeline: could not download %s for case %s: %s",
                             doc_type.value, case_id, exc)
                return doc, None

        # ── 1. OCR ─────────────────────────────────────────────
        aadhaar_doc, aadhaar_bytes = _get_doc_bytes(DocumentType.AADHAAR)
        pan_doc, pan_bytes = _get_doc_bytes(DocumentType.PAN)

        if aadhaar_doc and aadhaar_bytes:
            try:
                run_and_persist_ocr(db, aadhaar_doc.id, aadhaar_bytes)
            except OcrServiceError as exc:
                logger.warning("OCR failed for Aadhaar on case %s: %s", case_id, exc)

        if pan_doc and pan_bytes:
            try:
                run_and_persist_ocr(db, pan_doc.id, pan_bytes)
            except OcrServiceError as exc:
                logger.warning("OCR failed for PAN on case %s: %s", case_id, exc)

        # ── 2. Aadhaar tampering ────────────────────────────────
        if aadhaar_doc and aadhaar_bytes:
            run_and_persist_aadhaar_tampering_check(db, aadhaar_doc.id, aadhaar_bytes)

        # ── 3. PAN tampering ───────────────────────────────────
        if pan_doc and pan_bytes:
            run_and_persist_pan_tampering_check(db, pan_doc.id, pan_bytes)

        # ── 4. Cross-document verification ─────────────────────
        try:
            run_cross_document_verification(db, case_uuid)
        except Exception as exc:
            logger.warning("Cross-doc verification failed for case %s: %s", case_id, exc)

        # ── 5. Face verification ───────────────────────────────
        selfie_doc, selfie_bytes = _get_doc_bytes(DocumentType.SELFIE)

        document_bytes_by_id = {}

        if aadhaar_doc and aadhaar_bytes:
            document_bytes_by_id[aadhaar_doc.id] = aadhaar_bytes

        if pan_doc and pan_bytes:
            document_bytes_by_id[pan_doc.id] = pan_bytes

        if selfie_doc and selfie_bytes:
            document_bytes_by_id[selfie_doc.id] = selfie_bytes

        try:
            run_face_verification_for_case(
                db=db,
                kyc_case_id=case_uuid,
                document_bytes_by_id=document_bytes_by_id,
            )
        except Exception as exc:
            logger.warning(
                "Face verification failed for case %s: %s",
                case_id,
                exc,
            )

        # ── 6. Liveness — requires a separately uploaded video ──
        # Not triggered here. Cases without a liveness check will
        # correctly degrade to INCONCLUSIVE in the fraud scorer.

        # ── 7. Fraud risk scoring ──────────────────────────────
        fraud_score = run_and_persist_fraud_risk_scoring(db, case_uuid)

        # ── 8. Update case status ─────────────────────────────
        if fraud_score.requires_manual_review or fraud_score.risk_level == RiskLevel.HIGH:
            case.status = KycStatus.REVIEW_REQUIRED
        else:
            case.status = KycStatus.APPROVED

        db.commit()
        logger.info(
            "Pipeline complete for case %s: status=%s risk=%s score=%d",
            case_id,
            case.status.value,
            fraud_score.risk_level.value,
            fraud_score.total_score,
        )

        return {
            "case_id": case_id,
            "status": case.status.value,
            "risk_level": fraud_score.risk_level.value,
            "total_score": fraud_score.total_score,
        }

    except Exception as exc:
        logger.exception("Pipeline failed for case %s: %s", case_id, exc)
        try:
            case = db.get(KycCase, case_uuid)
            if case:
                case.status = KycStatus.REVIEW_REQUIRED
                db.commit()
        except Exception:
            pass
        raise self.retry(exc=exc)

    finally:
        db.close()
