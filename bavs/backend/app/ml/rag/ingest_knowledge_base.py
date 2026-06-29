#!/usr/bin/env python3
"""
Knowledge base ingestion CLI.

Run this ONCE after placing your KYC Rules, RBI Guidelines, Loan
Policy, and FAQ PDFs in backend/knowledge_base/pdfs/. Re-run whenever
you add or update PDFs to rebuild and re-upload the FAISS index.

USAGE (from backend/ directory):
    python -m app.ml.rag.ingest_knowledge_base

    # With custom chunk settings:
    python -m app.ml.rag.ingest_knowledge_base --chunk-size 600 --overlap 75

    # Dry run (validate PDFs without building or uploading index):
    python -m app.ml.rag.ingest_knowledge_base --dry-run

PREREQUISITES:
    1. PDFs placed in backend/knowledge_base/pdfs/
    2. R2 credentials set in .env:
       R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
       R2_BUCKET_NAME, R2_ENDPOINT_URL
    3. pip install -r requirements.txt (sentence-transformers, faiss-cpu,
       langchain, langchain-community, langchain-text-splitters)

NOTE: This script downloads the SentenceTransformer model weights on
first run (all-MiniLM-L6-v2, ~90MB). Subsequent runs use the local
cache. Run in Docker/local where network access to HuggingFace Hub is
available.
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and upload the FAISS knowledge base index to R2."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum characters per chunk (default: 500).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Character overlap between adjacent chunks (default: 50).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and load PDFs, count chunks, but do not build "
            "the FAISS index or upload to R2. Use this to check your "
            "PDFs are readable before committing to a full ingestion run."
        ),
    )
    args = parser.parse_args()

    # Late import so config validation (DATABASE_URL required) only fires
    # when the script actually runs, not at import time during tests.
    from app.ml.rag.pdf_loader import load_all_pdfs
    from app.ml.rag.chunking import chunk_documents
    from app.ml.rag.faiss_index import build_and_upload_index
    from app.services.storage_service import StorageServiceError

    logger.info("Loading PDFs from knowledge_base/pdfs/...")
    try:
        documents = load_all_pdfs()
    except (FileNotFoundError, Exception) as exc:
        logger.error("PDF loading failed: %s", exc)
        sys.exit(1)

    chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.overlap)
    logger.info(
        "Loaded %d page-documents from %d unique source files, "
        "producing %d chunks.",
        len(documents),
        len({d.metadata["source"] for d in documents}),
        len(chunks),
    )

    if args.dry_run:
        logger.info(
            "Dry run complete. No index built or uploaded. "
            "Re-run without --dry-run to build and upload."
        )
        return

    logger.info("Building FAISS index and uploading to R2...")
    try:
        vectorstore = build_and_upload_index(
            chunk_size=args.chunk_size, chunk_overlap=args.overlap
        )
    except StorageServiceError as exc:
        logger.error("R2 upload failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Index build failed: %s", exc)
        sys.exit(1)

    # Sanity check: run a test query against the just-built index to
    # confirm retrieval works end-to-end before declaring ingestion
    # complete.
    logger.info("Running sanity-check retrieval query...")
    results = vectorstore.similarity_search("KYC document requirements", k=3)
    if results:
        logger.info(
            "Sanity check passed: retrieved %d chunks for test query. "
            "First result source: %s (page %s).",
            len(results),
            results[0].metadata.get("source", "?"),
            results[0].metadata.get("page", "?"),
        )
    else:
        logger.warning(
            "Sanity check: no results returned for test query. "
            "The index was uploaded but may be empty or corrupt. "
            "Check your PDFs and re-run ingestion."
        )

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
