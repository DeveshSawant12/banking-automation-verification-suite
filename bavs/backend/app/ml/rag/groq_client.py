"""
Groq LLM client for RAG answer generation.

Uses the official `groq` Python SDK (not a fabricated API — Groq
publishes this SDK at https://github.com/groq/groq-python). The model
ID is read from settings.GROQ_MODEL_NAME (env var GROQ_MODEL_NAME),
defaulting to llama-3.3-70b-versatile per the locked decision.

DEPRECATION WARNING baked into the code: llama-3.3-70b-versatile is
scheduled for Groq shutdown on August 16, 2026. A startup-time check
logs this warning clearly whenever the deprecated model is configured,
so it's visible in deployment logs without anyone needing to re-read
this source file.

SYSTEM PROMPT DESIGN: the system prompt is deliberately narrow in scope
("you are a banking assistant, answer ONLY from the provided context")
rather than a generic assistant prompt. This is intentional for a
banking application — the RAG system should not generate answers from
the LLM's general training data, only from the retrieved KYC/RBI/Loan
Policy chunks. Answers that cannot be supported by the retrieved context
should be declined, not hallucinated.
"""

from __future__ import annotations

import logging

from groq import Groq

from app.config import get_settings

logger = logging.getLogger(__name__)

DEPRECATED_MODELS = {
    "llama-3.3-70b-versatile": "August 16, 2026",
    "llama-3.1-8b-instant": "August 16, 2026",
}

SYSTEM_PROMPT = """You are a banking compliance and KYC assistant for a financial institution. \
Your role is to answer questions about KYC procedures, RBI guidelines, loan policies, \
and banking regulations using ONLY the information provided in the context below.

Rules you must follow:
1. Answer ONLY from the provided context. Do not use general knowledge or training data.
2. If the context does not contain enough information to answer the question, say so explicitly: \
"I cannot find sufficient information about this in the available policy documents."
3. Always cite the source document and page number when available in the context metadata.
4. Do not provide legal or financial advice. Refer complex cases to a human compliance officer.
5. Be concise and factual. Avoid speculative language."""


def _check_model_deprecation(model_name: str) -> None:
    if model_name in DEPRECATED_MODELS:
        shutdown_date = DEPRECATED_MODELS[model_name]
        logger.warning(
            "DEPRECATION WARNING: Groq model '%s' is scheduled for "
            "shutdown on %s. Update the GROQ_MODEL_NAME environment "
            "variable before that date to avoid service interruption. "
            "See https://console.groq.com/docs/deprecations for "
            "recommended replacements.",
            model_name,
            shutdown_date,
        )


class GroqClientError(Exception):
    """Raised when the Groq API call fails or returns an unusable response."""


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """
    Generate an answer to the user's question using retrieved context
    chunks and the Groq LLM.

    Args:
        question: the user's natural-language question
        retrieved_chunks: list of dicts, each with keys:
            'content' (str), 'source' (str), 'page' (int)
            These come from rag_pipeline.retrieve_relevant_chunks().
        max_tokens: max tokens in the LLM's response
        temperature: sampling temperature (0.1 = low randomness,
            appropriate for factual regulatory Q&A)

    Returns:
        The LLM's answer string.

    Raises:
        GroqClientError: on API failure, missing API key, or an
            empty/null response from the model.
    """
    settings = get_settings()

    if not settings.GROQ_API_KEY:
        raise GroqClientError(
            "GROQ_API_KEY is not set. Add it to your .env file before "
            "using the RAG chatbot."
        )

    _check_model_deprecation(settings.GROQ_MODEL_NAME)

    # Build context block from retrieved chunks, including source
    # attribution in the prompt so the model can cite them per the
    # system prompt's instruction.
    context_parts = []
    for chunk in retrieved_chunks:
        source_label = f"[Source: {chunk['source']}, page {chunk.get('page', '?')}]"
        context_parts.append(f"{source_label}\n{chunk['content']}")

    context_block = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Context from banking policy documents:\n\n"
        f"{context_block}\n\n"
        f"Question: {question}"
    )

    try:
        client = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        raise GroqClientError(
            f"Groq API call failed (model={settings.GROQ_MODEL_NAME}): {exc}"
        ) from exc

    if not response.choices or not response.choices[0].message.content:
        raise GroqClientError(
            f"Groq API returned an empty response for model "
            f"{settings.GROQ_MODEL_NAME}. Check the model ID and "
            f"API key validity."
        )

    return response.choices[0].message.content.strip()
