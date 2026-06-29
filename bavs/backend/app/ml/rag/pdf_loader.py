"""
PDF loader for the RAG knowledge base.

Loads PDFs from knowledge_base/pdfs/ using pypdf (already installed as a
transitive dependency) and returns LangChain Document objects carrying
both the raw text content and a metadata dict that preserves the source
filename, page number, and total page count — metadata that will be
stored alongside the FAISS index entries and later surfaced in
chat_messages.retrieved_chunks_json for explainability.

This module intentionally does NOT scan for PDFs recursively — only
files placed directly inside knowledge_base/pdfs/ are treated as
knowledge-base documents. Subdirectory isolation is left to the
operator, not auto-discovered, to prevent accidentally indexing
temporary or unreviewed files.

No PDF content is fabricated here — the entire point of the locked
decision to "supply PDFs yourself" is that this module loads real
documents you place in knowledge_base/pdfs/. If the directory is empty
at ingestion time, build_knowledge_base() in faiss_index.py will raise
an explicit, clear error rather than silently building a zero-document
index that appears to work but returns garbage for every query.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[4] / "knowledge_base" / "pdfs"


class PdfLoadError(Exception):
    """Raised when a PDF cannot be loaded or yields no text content."""


def load_pdf_as_documents(pdf_path: Path) -> list[Document]:
    """
    Load a single PDF file and return one LangChain Document per page.
    Pages with zero extracted text (e.g. scanned image pages without OCR)
    are skipped with a warning rather than silently included as empty
    chunks, since empty chunks degrade retrieval quality without adding
    information.

    Raises:
        PdfLoadError: if the file cannot be opened or if ALL pages are
            blank/unextractable (indicating a fully image-based PDF that
            needs OCR preprocessing before indexing).
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PdfLoadError(
            "pypdf is not installed. Add 'pypdf' to requirements.txt."
        ) from exc

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PdfLoadError(f"Could not open PDF {pdf_path.name}: {exc}") from exc

    total_pages = len(reader.pages)
    documents: list[Document] = []
    skipped_blank = 0

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "Could not extract text from page %d of %s: %s — skipping.",
                page_num,
                pdf_path.name,
                exc,
            )
            skipped_blank += 1
            continue

        text = text.strip()
        if not text:
            skipped_blank += 1
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_num,
                    "total_pages": total_pages,
                },
            )
        )

    if not documents:
        raise PdfLoadError(
            f"No extractable text found in {pdf_path.name} ({total_pages} pages, "
            f"{skipped_blank} blank/image-only). If this is a scanned PDF, run OCR "
            f"preprocessing before adding it to the knowledge base."
        )

    if skipped_blank > 0:
        logger.warning(
            "%s: skipped %d/%d blank or image-only pages.",
            pdf_path.name,
            skipped_blank,
            total_pages,
        )

    logger.info(
        "Loaded %s: %d pages extracted (%d skipped).",
        pdf_path.name,
        len(documents),
        skipped_blank,
    )
    return documents


def load_all_pdfs(knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR) -> list[Document]:
    """
    Load every PDF in the knowledge base directory and return a flat list
    of all per-page LangChain Documents across all files.

    Raises:
        FileNotFoundError: if the knowledge base directory does not exist.
        PdfLoadError: if the directory exists but contains no .pdf files
            at all (not just empty pages — literally no PDF files present).
    """
    if not knowledge_base_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {knowledge_base_dir}. "
            f"Create this directory and place your KYC/RBI/Loan Policy "
            f"PDF files inside it before running ingestion."
        )

    pdf_files = sorted(knowledge_base_dir.glob("*.pdf"))
    if not pdf_files:
        raise PdfLoadError(
            f"No .pdf files found in {knowledge_base_dir}. "
            f"Add your KYC Rules, RBI Guidelines, Loan Policies, and FAQ "
            f"PDFs to this directory before running ingestion."
        )

    all_documents: list[Document] = []
    failed: list[str] = []

    for pdf_path in pdf_files:
        try:
            docs = load_pdf_as_documents(pdf_path)
            all_documents.extend(docs)
        except PdfLoadError as exc:
            logger.error("Failed to load %s: %s — skipping.", pdf_path.name, exc)
            failed.append(pdf_path.name)

    if not all_documents:
        raise PdfLoadError(
            f"All {len(pdf_files)} PDFs in {knowledge_base_dir} failed to yield "
            f"extractable text. Failed files: {failed}. Cannot build an index "
            f"with zero documents."
        )

    if failed:
        logger.warning(
            "%d/%d PDFs failed to load and were skipped: %s",
            len(failed),
            len(pdf_files),
            failed,
        )

    logger.info(
        "Loaded %d documents from %d PDFs (%d failed).",
        len(all_documents),
        len(pdf_files),
        len(failed),
    )
    return all_documents
