"""
Pydantic schemas for Module 10 (RAG Banking Chatbot).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime


class ChatMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2000)


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    model_used: str | None
    created_at: datetime


class ChatAnswerResponse(BaseModel):
    """Response returned after a user sends a question — includes both
    the AI answer and the retrieved source chunks for transparency."""

    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse
    retrieved_chunks: list[dict]
