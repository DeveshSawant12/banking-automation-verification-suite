"""
Pydantic schemas for tampering detection (Aadhaar — Module 2, and PAN —
Module 3, which share this schema since the verdict structure is
identical for both document types).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models.tampering_result import TamperVerdict


class TamperingResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    verdict: TamperVerdict
    confidence: float
    ela_score: float
    gradcam_heatmap_r2_key: str | None
    model_version: str
    created_at: datetime


class TamperingAnalysisInternal(BaseModel):
    """
    Internal result shape returned by analyze_aadhaar_tampering /
    analyze_pan_tampering before persistence. Not exposed directly via
    API (the ela_image numpy array is not JSON-serializable) — kept
    separate from the DB-backed response schema above.
    """

    verdict: str
    confidence: float
    ela_score: float
    model_version: str
