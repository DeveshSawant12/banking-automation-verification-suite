"""
RAG pipeline — retrieval + generation orchestration.

Composes faiss_index.py (vector retrieval) and groq_client.py (LLM
answer generation) into a single callable that rag_service.py uses.

The FAISS vectorstore is cached as a module-level singleton — consistent
with the project's established pattern for heavy ML models (EasyOCR,
ResNet18, MediaPipe). The index is fetched from R2 on first call and
stays in memory for the process lifetime. This means:
  - A FastAPI process startup triggers one R2 download.
  - Re-ingestion (uploading a new index to R2) does NOT hot-reload the
    running process — the service must be restarted for a new index to
    take effect. This is an explicit, documented limitation, not a
    silent one. Hot-reloading would require a cache-busting mechanism
    (e.g. a version key in R2 + a background polling thread) which is
    beyond the current project scope.

TOP_K_CHUNKS controls how many retrieved chunks are passed to the LLM.
Too few → the LLM may not have enough context to answer. Too many → the
prompt grows toward the model's context limit and becomes slower/costlier.
5 is the commonly-published practical default for retrieval over policy-
scale document corpora with all-MiniLM-L6-v2.
"""

from __future__ import annotations

import logging
import threading

# NOTE: see faiss_index.py for note on langchain-community deprecation status.
from langchain_community.vectorstores import FAISS  # noqa: E402

from app.ml.rag.faiss_index import FaissIndexNotFoundError, load_index_from_r2
from app.ml.rag.groq_client import GroqClientError, generate_answer

logger = logging.getLogger(__name__)

TOP_K_CHUNKS = 5

_index_lock = threading.Lock()
_vectorstore: FAISS | None = None


def _get_vectorstore() -> FAISS:
    """
    Lazily load and cache the FAISS vectorstore from R2. Thread-safe.

    Raises:
        FaissIndexNotFoundError: if no index exists in R2 yet.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    with _index_lock:
        if _vectorstore is None:
            _vectorstore = load_index_from_r2()
    return _vectorstore


def retrieve_relevant_chunks(query: str, top_k: int = TOP_K_CHUNKS) -> list[dict]:
    """
    Retrieve the top-k most semantically relevant chunks from the FAISS
    index for the given query.

    Returns:
        List of dicts, each with:
            'content': str — the chunk's text
            'source': str — source PDF filename
            'page': int — page number within that PDF
            'chunk_index': int — chunk's position in the full corpus
            'similarity_score': float — L2 distance from FAISS (lower is
                more similar; included here for the
                retrieved_chunks_json audit field on chat_messages)

    Raises:
        FaissIndexNotFoundError: if no index has been built yet.
    """
    vectorstore = _get_vectorstore()

    results = vectorstore.similarity_search_with_score(query, k=top_k)

    chunks = []
    for doc, score in results:
        chunks.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "similarity_score": float(score),
            }
        )
    return chunks


def answer_question(query: str) -> dict:
    """
    Full RAG pipeline: retrieve → generate.

    Returns:
        dict with keys:
            'answer': str — the LLM's generated answer
            'retrieved_chunks': list[dict] — the chunks used as context
                (stored by rag_service.py in chat_messages.retrieved_chunks_json)
            'model_used': str — the actual Groq model ID that generated
                the answer (useful for the audit trail given the pending
                deprecation)

    Raises:
        FaissIndexNotFoundError: if no knowledge base has been ingested.
        GroqClientError: if the Groq API call fails.
    """
    from app.config import get_settings

    chunks = retrieve_relevant_chunks(query)

    if not chunks:
        return {
            "answer": (
                "No relevant information was found in the knowledge base "
                "for your question. Please contact a compliance officer for "
                "assistance."
            ),
            "retrieved_chunks": [],
            "model_used": get_settings().GROQ_MODEL_NAME,
        }

    answer = generate_answer(question=query, retrieved_chunks=chunks)

    return {
        "answer": answer,
        "retrieved_chunks": chunks,
        "model_used": get_settings().GROQ_MODEL_NAME,
    }
