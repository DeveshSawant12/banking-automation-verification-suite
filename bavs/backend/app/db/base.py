from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Import all models here so SQLAlchemy's mapper always sees the full
# relationship graph, regardless of which module imports Base first.
def _import_all_models():
    from app.db.models import user  # noqa: F401
    from app.db.models import kyc_case  # noqa: F401
    from app.db.models import document  # noqa: F401
    from app.db.models import ocr_extraction  # noqa: F401
    from app.db.models import tampering_result  # noqa: F401
    from app.db.models import face_verification_result  # noqa: F401
    from app.db.models import cross_document_result  # noqa: F401
    from app.db.models import liveness_result  # noqa: F401
    from app.db.models import fraud_risk_score  # noqa: F401
    from app.db.models import audit_log  # noqa: F401
    from app.db.models import chat  # noqa: F401


_import_all_models()