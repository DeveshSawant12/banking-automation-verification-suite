"""
SentenceTransformers embedding engine for the RAG pipeline.

Uses all-MiniLM-L6-v2 (locked decision), a 384-dimensional embedding
model. This module provides a singleton loader (consistent with the
project's pattern for heavy ML models — EasyOCR, ResNet18, MediaPipe)
and a LangChain-compatible embedding class that FAISS index building
(faiss_index.py) and query-time retrieval (rag_pipeline.py) both use,
so the same model instance serves both the ingestion CLI and the live
inference path without re-loading weights on every call.

LANGCHAIN INTEGRATION: MiniLMEmbeddings inherits from LangChain's
Embeddings base class (langchain_core.embeddings.Embeddings), which is
required for langchain_community's FAISS vectorstore to use
.embed_query() via isinstance check rather than treating the object as
a plain callable. This was established by inspecting the actual source
of FAISS._embed_query in the installed version (0.4.2).
"""

from __future__ import annotations

import logging
import threading
from typing import List

import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model_lock = threading.Lock()
_model_instance: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Lazily initialize and return the singleton SentenceTransformer model.
    Thread-safe for concurrent Celery worker access.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    with _model_lock:
        if _model_instance is None:
            logger.info("Loading SentenceTransformer: %s ...", EMBEDDING_MODEL_NAME)
            _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("SentenceTransformer loaded.")
    return _model_instance


class MiniLMEmbeddings(Embeddings):
    """
    LangChain Embeddings implementation backed by all-MiniLM-L6-v2.
    Inherits from langchain_core.embeddings.Embeddings so that
    langchain_community's FAISS vectorstore correctly identifies it
    via isinstance check and dispatches to .embed_query() /
    .embed_documents() rather than attempting to call it as a function.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings. Called by LangChain's
        FAISS.from_documents() during index building.
        """
        if not texts:
            return []
        model = get_embedding_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string. Called by LangChain's FAISS
        similarity_search_with_score() at query time.
        """
        model = get_embedding_model()
        embedding = model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding[0].tolist()
