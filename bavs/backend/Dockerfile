# ================================================================
# Backend Dockerfile
# Base: python:3.12-slim (matches the Python version used in dev)
# ================================================================

FROM python:3.12-slim AS base

# System packages needed by OpenCV, EasyOCR, PyTorch, and psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directories that must exist at runtime
RUN mkdir -p ml_models knowledge_base/pdfs knowledge_base/faiss_index training_data/raw

# ================================================================
# Development target: run with uvicorn --reload
# ================================================================
FROM base AS development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ================================================================
# Celery worker target
# ================================================================
FROM base AS celery-worker
CMD ["celery", "-A", "app.tasks.celery_worker", "worker", "--loglevel=info", "--concurrency=2"]

# ================================================================
# Production target: run with uvicorn (no reload, multiple workers)
# ================================================================
FROM base AS production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
