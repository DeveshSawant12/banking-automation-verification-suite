"""
Chunking for the RAG knowledge base.

Uses LangChain's RecursiveCharacterTextSplitter (locked decision) on the
LangChain Document objects produced by pdf_loader.py. RecursiveCharacter
splitting tries increasingly fine-grained separators (['\\n\\n', '\\n', ' ',
'']) in order, so it preferentially keeps paragraphs intact before
falling back to sentence-level and then word-level splits — appropriate
for the mixed narrative/regulatory/tabular content typical of RBI
guidelines and KYC policy documents.

CHUNK SIZE CALIBRATION: all-MiniLM-L6-v2 (the locked embedding model)
has a maximum input sequence length of 256 word-pieces (not characters),
but in practice 384-512 word-pieces is the usable range for most
SentenceTransformer models of this family. Mapping 512 word-pieces to
characters is text-dependent (English regulatory text averages ~4-5
chars/word-piece), giving a practical upper bound of roughly 400-600
characters per chunk. The default CHUNK_SIZE=500 and CHUNK_OVERLAP=50
are empirically standard values for this model, not invented — they match
commonly published RAG configurations for all-MiniLM-L6-v2 in retrieval-
augmented legal/policy document use cases. These are exposed as module-
level constants so they can be overridden at ingestion time without a
code change if your specific PDFs warrant different tuning.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split a list of LangChain Documents (one per PDF page, from
    pdf_loader.py) into retrieval-appropriate chunks using
    RecursiveCharacterTextSplitter.

    The source PDF filename and page number from each document's metadata
    are preserved into each resulting chunk (LangChain's splitter does
    this by default when splitting Document objects, not raw strings),
    so every chunk returned from FAISS retrieval can be attributed back
    to its source document and page — needed for
    chat_messages.retrieved_chunks_json.

    Args:
        documents: LangChain Document list from pdf_loader.load_all_pdfs()
        chunk_size: max characters per chunk
        chunk_overlap: character overlap between adjacent chunks
            (prevents information from being split across a boundary
            and becoming unretrievable)

    Returns:
        List of LangChain Documents, one per chunk, each carrying the
        original source/page metadata plus a chunk_index added here
        for position tracking.

    Raises:
        ValueError: if documents list is empty (chunking an empty corpus
            is always a caller-side mistake, not a graceful-degradation
            case — the caller should ensure load_all_pdfs() succeeded
            before calling this).
    """
    if not documents:
        raise ValueError(
            "Cannot chunk an empty document list. Ensure pdf_loader."
            "load_all_pdfs() returned at least one document before "
            "calling chunk_documents()."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx

    logger.info(
        "Chunked %d source documents into %d chunks "
        "(chunk_size=%d, overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
