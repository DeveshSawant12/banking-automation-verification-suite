"""Aggregates all v1 routers into a single include-ready APIRouter."""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.kyc import router as kyc_router
from app.api.v1.verification import router as verification_router
from app.api.v1.other_routers import (
    fraud_router,
    audit_router,
    chatbot_router,
    admin_router,
)

# No prefix here -- prefix is applied in main.py via include_router(prefix=...)
# so that the routes from each sub-router (which already carry their own prefix,
# e.g. /auth, /kyc) are correctly mounted at /api/v1/auth, /api/v1/kyc, etc.
v1_router = APIRouter()

v1_router.include_router(auth_router)
v1_router.include_router(kyc_router)
v1_router.include_router(verification_router)
v1_router.include_router(fraud_router)
v1_router.include_router(audit_router)
v1_router.include_router(chatbot_router)
v1_router.include_router(admin_router)
