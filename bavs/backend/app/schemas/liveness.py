"""
Pydantic schemas for Liveness Detection (Module 6).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models.liveness_result import LivenessVerdict


class LivenessResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kyc_case_id: uuid.UUID
    verdict: LivenessVerdict
    blink_detected: bool
    head_movement_detected: bool
    smile_detected: bool
    max_yaw_delta_degrees: float
    total_frames_analyzed: int
    frames_with_face: int
    frames_without_face: int
    created_at: datetime


class LivenessAnalysisInternal(BaseModel):
    """
    Internal result shape returned by liveness_service before persistence.
    Mirrors TamperingAnalysisInternal's role from Module 2/3.
    """

    verdict: str
    blink_detected: bool
    head_movement_detected: bool
    smile_detected: bool
    max_yaw_delta_degrees: float
    total_frames_analyzed: int
    frames_with_face: int
    frames_without_face: int
