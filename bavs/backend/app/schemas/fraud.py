"""
Pydantic schemas for the Fraud Risk Engine (Module 7).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models.fraud_risk_score import RiskLevel


class FraudRiskScoreResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kyc_case_id: uuid.UUID
    total_score: int
    risk_level: RiskLevel
    score_breakdown: dict
    requires_manual_review: bool
    created_at: datetime


class FraudRiskAnalysisInternal(BaseModel):
    """
    Internal result shape returned by fraud_risk_service before
    persistence. total_score is clamped to [0, 100] for risk-level
    banding purposes (see fraud_risk_service docstring on why raw
    accumulated points can exceed 100 but the persisted total_score is
    capped).
    """

    total_score: int
    risk_level: str
    score_breakdown: dict
    requires_manual_review: bool
