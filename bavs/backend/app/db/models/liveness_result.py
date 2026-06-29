"""
LivenessResult model — one row per kyc_case_id, storing the aggregated
verdict from video_sequence_analyzer.analyze_frame_sequence (Module 6).

Matches the original locked schema (Module 6: LIVE / SPOOF / INCONCLUSIVE,
blink/head-movement/smile booleans, raw_metrics_json) with one addition
beyond the original spec: max_yaw_delta_degrees is stored as its own
column (not just buried in raw_metrics_json), because the Fraud Risk
Engine (Module 7) and Admin Dashboard (Module 11) need to query/display
the magnitude of head movement without parsing JSON — this is an
additive column, not a removal or rename of anything originally
specified.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class LivenessVerdict(str, enum.Enum):
    LIVE = "LIVE"
    SPOOF = "SPOOF"
    INCONCLUSIVE = "INCONCLUSIVE"


class LivenessResult(Base):
    __tablename__ = "liveness_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kyc_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("kyc_cases.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    verdict: Mapped[LivenessVerdict] = mapped_column(
        Enum(LivenessVerdict, name="liveness_verdict"), nullable=False
    )
    blink_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    head_movement_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    smile_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    max_yaw_delta_degrees: Mapped[float] = mapped_column(Float, nullable=False)
    total_frames_analyzed: Mapped[int] = mapped_column(Integer, nullable=False)
    frames_with_face: Mapped[int] = mapped_column(Integer, nullable=False)
    frames_without_face: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_metrics_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    kyc_case: Mapped["KycCase"] = relationship("KycCase")
