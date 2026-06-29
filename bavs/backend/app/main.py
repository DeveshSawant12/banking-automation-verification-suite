"""
FastAPI application entrypoint.

This file creates the FastAPI app, mounts every router, and configures
CORS. It is what uvicorn imports when it starts the backend:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import v1_router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Banking Automation & Verification Suite",
    description=(
        "AI-powered KYC verification platform: OCR, tampering detection, "
        "face verification, liveness detection, fraud risk scoring, "
        "explainable AI, audit logs, and RAG banking chatbot."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────
# settings.CORS_ORIGINS is a list[str] from .env, e.g.
# ["http://localhost:5173", "https://your-frontend.example.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────
app.include_router(v1_router, prefix="/api/v1")


# ── Health check ─────────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health():
    """Used by Docker Compose and Render health checks."""
    return {"status": "ok"}
