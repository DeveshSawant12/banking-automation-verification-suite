"""
RAG Service — Module 10 orchestration layer.

Manages chat session/message lifecycle (DB persistence) and delegates
the actual retrieval+generation to rag_pipeline.answer_question().

ERROR HANDLING: if the FAISS index has not been ingested yet
(FaissIndexNotFoundError), the service does NOT crash — it persists the
user's message and returns a clear "knowledge base not available yet"
assistant message. This lets the API remain operational even when the
index hasn't been built, which is expected during initial deployment
before the operator runs the ingestion CLI.

Audit logging is wired in at the session-creation level only (not per-
message), consistent with the AuditEventType taxonomy — chatbot usage is
not listed as a per-message auditable event in the spec, only the broader
verification events are. This can be extended later without modifying the
existing audit_service.py contract.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.orm import Session

from app.db.models.chat import ChatMessage, ChatSession
from app.db.models.user import User
from app.ml.rag.faiss_index import FaissIndexNotFoundError
from app.ml.rag.groq_client import GroqClientError
from app.ml.rag.rag_pipeline import answer_question

logger = logging.getLogger(__name__)

INDEX_NOT_READY_MESSAGE = (
    "The banking knowledge base is not yet available. The system "
    "administrator needs to run the ingestion pipeline first. "
    "Please contact your bank's support team for assistance."
)

GROQ_ERROR_MESSAGE = (
    "The AI assistant is temporarily unavailable. Please try again "
    "shortly or contact your bank's support team for assistance."
)


class RagServiceError(Exception):
    """Raised on unrecoverable errors in the RAG service layer."""


def create_chat_session(db: Session, user_id: uuid.UUID) -> ChatSession:
    """Create and persist a new chat session for the given user."""
    user = db.get(User, user_id)
    if user is None:
        raise RagServiceError(f"User {user_id} does not exist.")

    session = ChatSession(user_id=user_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info("Created chat session %s for user %s", session.id, user_id)
    return session


def get_session_messages(
    db: Session, session_id: uuid.UUID, user_id: uuid.UUID
) -> list[ChatMessage]:
    """
    Retrieve all messages in a session. Validates that the requesting
    user owns the session (per the "any authenticated user can query but
    only their own sessions" security model).
    """
    session = db.get(ChatSession, session_id)
    if session is None:
        raise RagServiceError(f"Chat session {session_id} does not exist.")
    if session.user_id != user_id:
        raise RagServiceError(
            f"User {user_id} does not own session {session_id}."
        )
    return session.messages


def send_message(
    db: Session,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    user_content: str,
) -> dict:
    """
    Process a user message through the full RAG pipeline and persist
    both the user message and the assistant's response.

    Returns:
        dict with keys: user_message (ChatMessage), assistant_message
        (ChatMessage), retrieved_chunks (list[dict])
    """
    session = db.get(ChatSession, session_id)
    if session is None:
        raise RagServiceError(f"Chat session {session_id} does not exist.")
    if session.user_id != user_id:
        raise RagServiceError(
            f"User {user_id} does not own session {session_id}."
        )

    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=user_content,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    retrieved_chunks: list[dict] = []
    answer_content: str
    model_used: str | None = None

    try:
        result = answer_question(user_content)
        answer_content = result["answer"]
        retrieved_chunks = result["retrieved_chunks"]
        model_used = result["model_used"]

    except FaissIndexNotFoundError as exc:
        logger.warning(
            "FAISS index not ready for session %s: %s", session_id, exc
        )
        answer_content = INDEX_NOT_READY_MESSAGE

    except GroqClientError as exc:
        logger.error(
            "Groq API error for session %s: %s", session_id, exc
        )
        answer_content = GROQ_ERROR_MESSAGE

    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=answer_content,
        retrieved_chunks_json=(
            {"chunks": retrieved_chunks} if retrieved_chunks else None
        ),
        model_used=model_used,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return {
        "user_message": user_msg,
        "assistant_message": assistant_msg,
        "retrieved_chunks": retrieved_chunks,
    }
