"""
FAISS vector index — build, persist to R2, and load.

Locked architecture: the FAISS index is persisted to Cloudflare R2
(consistent with all other project artifacts: uploaded documents,
GradCAM heatmaps, feature vectors). This means:
  - The ingestion CLI (build_knowledge_base below) builds the index
    locally from your PDFs, then uploads the serialized index to R2.
  - The live RAG pipeline (rag_pipeline.py) downloads the index from R2
    at startup and caches it in memory for the process lifetime.
  - Local disk is used only as a transient staging path during
    build/load — not as the durable store.

R2 keys for the FAISS index follow a fixed naming convention so that
the upload path (ingestion CLI) and the download path (live app) always
agree without configuration:
    faiss_index/index.faiss
    faiss_index/index.pkl   (LangChain's docstore metadata companion)

These are REPLACED on every re-ingestion, since FAISS does not support
incremental updates without recomputing the full index (a real FAISS
limitation, not an invented one). For a corpus of the size expected
(KYC/RBI/Loan Policy docs), full re-ingestion is fast enough to not
warrant a more complex incremental strategy.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

# NOTE: langchain-community is marked for deprecation as of 2025/2026,
# but the standalone 'langchain-faiss' replacement package is not yet
# published to PyPI (verified June 2026). This import remains correct
# for the installed version (0.4.2). Revisit when 'langchain-faiss'
# becomes available as a standalone pip install.
from langchain_community.vectorstores import FAISS  # noqa: E402

from app.ml.rag.chunking import chunk_documents
from app.ml.rag.embeddings import MiniLMEmbeddings
from app.ml.rag.pdf_loader import load_all_pdfs
from app.services.storage_service import StorageServiceError, download_bytes, upload_bytes

logger = logging.getLogger(__name__)

FAISS_INDEX_R2_KEY = "faiss_index/index.faiss"
FAISS_PKL_R2_KEY = "faiss_index/index.pkl"
LOCAL_INDEX_FILENAME = "index.faiss"
LOCAL_PKL_FILENAME = "index.pkl"


class FaissIndexNotFoundError(Exception):
    """Raised when the FAISS index cannot be found in R2 storage."""


def build_and_upload_index(
    chunk_size: int = 500, chunk_overlap: int = 50
) -> FAISS:
    """
    Full ingestion pipeline: load all PDFs → chunk → embed → build FAISS
    index → upload to R2.

    Intended to be called from the CLI ingestion script
    (ingest_knowledge_base.py), not from live request handlers. Returns
    the built FAISS object so the caller can optionally run a quick
    sanity-check query against it before it is relied upon for live
    traffic.

    Raises:
        PdfLoadError: if no usable PDFs exist in knowledge_base/pdfs/
        StorageServiceError: if the upload to R2 fails
    """
    logger.info("Starting knowledge base ingestion...")
    documents = load_all_pdfs()
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info("Building FAISS index from %d chunks...", len(chunks))
    embeddings = MiniLMEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        vectorstore.save_local(str(tmp_path))

        index_bytes = (tmp_path / LOCAL_INDEX_FILENAME).read_bytes()
        pkl_bytes = (tmp_path / LOCAL_PKL_FILENAME).read_bytes()

    try:
        upload_bytes(index_bytes, key_prefix="faiss_index", content_type="application/octet-stream")
        upload_bytes(pkl_bytes, key_prefix="faiss_index", content_type="application/octet-stream")
    except StorageServiceError as exc:
        raise StorageServiceError(
            f"FAISS index built successfully but upload to R2 failed: {exc}. "
            f"Re-run ingestion to retry the upload."
        ) from exc

    logger.info(
        "FAISS index uploaded to R2 (%d chunks, %d source documents).",
        len(chunks),
        len(documents),
    )
    return vectorstore


def load_index_from_r2() -> FAISS:
    """
    Download the FAISS index from R2 and load it into memory. Called once
    at application startup by rag_service.py; the returned FAISS object
    is cached as a module-level singleton there.

    Raises:
        FaissIndexNotFoundError: if the index does not exist in R2 (i.e.
            ingestion has not been run yet). The RAG service must catch
            this and return a clear error to the API rather than crashing.
        StorageServiceError: on R2 connectivity failures.
    """
    try:
        index_bytes = download_bytes(FAISS_INDEX_R2_KEY)
        pkl_bytes = download_bytes(FAISS_PKL_R2_KEY)
    except StorageServiceError as exc:
        if "NoSuchKey" in str(exc) or "404" in str(exc):
            raise FaissIndexNotFoundError(
                "FAISS index not found in R2. Run the ingestion script "
                "(python -m app.ml.rag.ingest_knowledge_base) after "
                "placing PDFs in knowledge_base/pdfs/ to build and "
                "upload the index."
            ) from exc
        raise

    embeddings = MiniLMEmbeddings()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / LOCAL_INDEX_FILENAME).write_bytes(index_bytes)
        (tmp_path / LOCAL_PKL_FILENAME).write_bytes(pkl_bytes)
        vectorstore = FAISS.load_local(
            str(tmp_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("FAISS index loaded from R2.")
    return vectorstore
