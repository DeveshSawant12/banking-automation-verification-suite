"""
Centralized application configuration.

All values are loaded from environment variables (.env in local dev,
Render environment variables in production). NOTHING is hardcoded here —
per project rules, no invented API keys or endpoints. Each field below
corresponds to a real, already-agreed-upon infrastructure dependency:
PostgreSQL (DB), Redis (Celery broker), Cloudflare R2 (object storage),
Groq (LLM for RAG, added in Module 10), JWT secret (auth).

Only the fields required by Module 1 (DB connection) are functionally
exercised so far; the rest are declared now because db/base.py and
storage/service imports will need them shortly and redefining this file
per-module would violate the "never rewrite unrelated files" rule.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database ---
    DATABASE_URL: str

    # --- Auth ---
    JWT_SECRET_KEY: str = "CHANGE_ME_IN_ENV"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # --- Redis / Celery ---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Cloudflare R2 (S3-compatible) ---
    R2_ACCOUNT_ID: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = ""
    R2_ENDPOINT_URL: str = ""

    # --- Groq (RAG, Module 10) ---
    GROQ_API_KEY: str = ""
    # llama-3.3-70b-versatile matches the spec's 'Groq Llama-3' requirement.
    # DEPRECATION WARNING: Groq has announced this model will be shut down
    # on August 16, 2026. Update this env var before that date to avoid
    # service interruption -- no code change required, only the env var.
    # Groq's own recommended replacement as of June 2026: openai/gpt-oss-120b
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"

    # --- App ---
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
