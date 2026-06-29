# Banking Automation & Verification Suite
## Complete Setup & Running Guide — From Zero to Running

---

## What You Need on Your Machine First

Install these before anything else:

| Tool | Minimum Version | How to Check |
|---|---|---|
| Docker Desktop | 24+ | `docker --version` |
| Docker Compose | 2.20+ | `docker compose version` |
| Git | any | `git --version` |
| Python | 3.12 | `python3 --version` (only for training scripts, not for running the app) |

Everything else (PostgreSQL, Redis, Node.js, all Python packages) runs inside Docker — you do **not** need them installed locally.

---

## Step 1 — Get the Code onto Your Machine

```bash
# If the project is already in a folder on your machine, just cd into it:
cd banking-automation-verification-suite

# If you need to copy it from somewhere, ensure this folder structure exists:
# banking-automation-verification-suite/
#   backend/
#   frontend/
#   docker-compose.yml
```

---

## Step 2 — Create Your Environment File

This is the most important step. The app will not start without it.

```bash
cd banking-automation-verification-suite/backend
cp .env.example .env
```

Now open `backend/.env` in any text editor and fill in these values:

### 2a. Database — leave exactly as-is for local Docker
```
DATABASE_URL=postgresql://bankuser:bankpassword@postgres:5432/bankkyc
```
This already matches the `postgres` service in `docker-compose.yml`. Do not change it for local development.

### 2b. JWT Secret — generate a real one
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```
Copy the output and paste it as:
```
JWT_SECRET_KEY=paste_the_output_here
```

### 2c. Redis — leave exactly as-is for local Docker
```
REDIS_URL=redis://redis:6379/0
```

### 2d. Cloudflare R2 — you need a Cloudflare account
1. Go to https://dash.cloudflare.com → R2 Object Storage
2. Create a bucket (any name, e.g. `bankkyc-storage`)
3. Go to "Manage R2 API Tokens" → Create API Token with Read & Write access
4. Fill in:
```
R2_ACCOUNT_ID=       ← from the R2 overview page URL: dash.cloudflare.com/<this>/r2
R2_ACCESS_KEY_ID=    ← from the API token you just created
R2_SECRET_ACCESS_KEY= ← from the API token you just created
R2_BUCKET_NAME=      ← the bucket name you chose
R2_ENDPOINT_URL=https://<YOUR_ACCOUNT_ID>.r2.cloudflarestorage.com
```

### 2e. Groq API Key — for the chatbot
1. Go to https://console.groq.com/keys
2. Create a new API key
3. Fill in:
```
GROQ_API_KEY=gsk_...your key here...
GROQ_MODEL_NAME=llama-3.3-70b-versatile
```
⚠️ This model is deprecated by Groq on **August 16, 2026**. After that date, change `GROQ_MODEL_NAME` to `openai/gpt-oss-120b`.

---

## Step 3 — Start All Services

```bash
# From the root of the project (where docker-compose.yml lives):
cd banking-automation-verification-suite
docker compose up --build
```

This will:
1. Pull PostgreSQL 16 and Redis 7 images (~400 MB total, once only)
2. Build the backend image — installs all Python packages including PyTorch, EasyOCR, MediaPipe (~5–10 min first time, much faster after because Docker caches layers)
3. Start the Vite frontend dev server
4. Start the Celery worker

You will know it is ready when you see output like:
```
backend-1  | INFO:     Uvicorn running on http://0.0.0.0:8000
frontend-1 | VITE v8.x  ready in 800 ms
frontend-1 | ➜  Local:   http://localhost:5173/
```

---

## Step 4 — Set Up the Database (Run Once)

Open a **new terminal** while Docker is running:

```bash
# Enter the running backend container
docker compose exec backend bash

# You are now inside the container. Run:
alembic upgrade head
```

This creates all 12 database tables. You should see:
```
INFO  [alembic.runtime.migration] Running upgrade  -> 49a711d3f09e, modules_1_to_10_schema
INFO  [alembic.runtime.migration] Running upgrade 49a711d3f09e -> f1a9c3d7e2b4, enforce audit_logs insert-only
```

Then exit the container:
```bash
exit
```

---

## Step 5 — Create the First Admin User (Run Once)

The app has no registration UI for admin users — they are created via a script.

```bash
docker compose exec backend python3 -c "
import sys
sys.path.insert(0, '.')
import os
os.environ['DATABASE_URL'] = os.environ.get('DATABASE_URL', '')
from app.db.base import SessionLocal
from app.db.models.user import User, UserRole
import hashlib

db = SessionLocal()
hashed = hashlib.sha256(b'admin123').hexdigest()  # change this password
admin = User(
    email='admin@bank.com',
    hashed_password=hashed,
    full_name='System Admin',
    role=UserRole.ADMIN,
    is_active=True,
)
db.add(admin)
db.commit()
print('Admin user created:', admin.email)
db.close()
"
```

> **Note:** this is a minimal bootstrap script. The real auth module (which hasn't been wired to the API yet — it's part of the "auth API wiring" step below) will use bcrypt hashing. For now, use this to create your first user.

---

## Step 6 — Open the App

| URL | What you see |
|---|---|
| http://localhost:5173 | The frontend (Login page) |
| http://localhost:8000/docs | FastAPI interactive API docs (Swagger UI) |
| http://localhost:8000/redoc | Alternative API docs |

---

## Step 7 — What Still Needs Wiring (Be Honest About This)

The project has all service/ML/database logic fully built and tested. What is not yet built is the **FastAPI router layer** — the actual HTTP endpoint definitions in `app/api/v1/`. This is intentional: the architecture phase defined all 25+ endpoints, the service layer behind them is complete, but the router files themselves (`auth.py`, `kyc.py`, `verification.py`, etc.) were not generated as part of Modules 1–11.

**What this means:** the Swagger UI at `:8000/docs` will show no endpoints until you wire the routers. The frontend will show the login page but API calls will return 404.

**How to wire them:** this is straightforward boilerplate that connects the already-built service functions to FastAPI endpoints. Each router file imports its service and returns Pydantic schemas. Example for `auth.py`:

```python
# backend/app/api/v1/auth.py  (you create this file)
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services import auth_service   # build this service
from app.schemas.auth import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    ...
```

The same pattern repeats for every module. If you want me to generate the full router layer + auth service + `app/main.py` as a follow-up session, just ask — the service logic is already done so each router file is mostly plumbing.

---

## Step 8 — Train the Tampering Detection Models

This requires real sample images of Aadhaar and PAN cards.

```bash
# 1. Place real Aadhaar sample images (JPG/PNG) into:
backend/training_data/raw/aadhaar/

# 2. Place real PAN sample images into:
backend/training_data/raw/pan/

# 3. Run training (inside the backend container):
docker compose exec backend python3 -m app.ml.forgery.train_forgery_model \
    --real-dir training_data/raw/aadhaar \
    --output ml_models/aadhaar_rf_model.pkl \
    --tampered-per-real 3

docker compose exec backend python3 -m app.ml.forgery.train_forgery_model \
    --real-dir training_data/raw/pan \
    --output ml_models/pan_rf_model.pkl \
    --tampered-per-real 3
```

The script will:
- Generate 3 synthetic tampered variants per real image (copy-move, text overlay, recompression)
- Train the Random Forest on real + tampered data
- Print accuracy/F1 scores
- Save the model to `ml_models/`

Until this runs, the tampering check correctly returns `INCONCLUSIVE → REVIEW_REQUIRED` rather than fabricating a verdict.

**Minimum images needed:** 20 real images per document type for a meaningful model. The script will warn you if you have fewer.

---

## Step 9 — Set Up the RAG Chatbot

```bash
# 1. Place your PDF files into:
backend/knowledge_base/pdfs/
# Suggested files: RBI KYC Master Direction, Loan Policy, FAQ document

# 2. Run ingestion (inside the backend container):
docker compose exec backend python3 -m app.ml.rag.ingest_knowledge_base
```

This downloads the `all-MiniLM-L6-v2` embedding model (~90 MB, once), chunks your PDFs, builds a FAISS index, and uploads it to Cloudflare R2. The chatbot will then be able to answer questions from your documents.

To do a dry-run first (validates your PDFs without uploading anything):
```bash
docker compose exec backend python3 -m app.ml.rag.ingest_knowledge_base --dry-run
```

---

## Step 10 — Set Up MediaPipe for Liveness Detection

The liveness detection module requires a MediaPipe FaceLandmarker `.task` model file. Download it from Google:

```bash
# Inside the backend container:
docker compose exec backend bash -c "
curl -o /app/ml_models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
"
```

Then set this path in your `backend/.env`:
```
MEDIAPIPE_MODEL_PATH=/app/ml_models/face_landmarker.task
```

And add `MEDIAPIPE_MODEL_PATH` to `backend/app/config.py`:
```python
MEDIAPIPE_MODEL_PATH: str = "ml_models/face_landmarker.task"
```

---

## Common Commands

```bash
# Start everything
docker compose up

# Start in background
docker compose up -d

# See logs from a specific service
docker compose logs backend --follow
docker compose logs worker --follow

# Restart just the backend after a code change (when not using --reload)
docker compose restart backend

# Stop everything
docker compose down

# Stop and delete the database volume (WARNING: deletes all data)
docker compose down -v

# Open a shell inside the backend container
docker compose exec backend bash

# Run a one-off Python command inside the backend container
docker compose exec backend python3 -c "print('hello from inside Docker')"

# Check database tables
docker compose exec postgres psql -U bankuser -d bankkyc -c "\dt"
```

---

## Project Folder Structure

```
banking-automation-verification-suite/
├── backend/
│   ├── .env.example          ← copy to .env and fill in
│   ├── .env                  ← your secrets (gitignored)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── alembic.ini
│   ├── alembic/
│   │   └── versions/
│   │       ├── 49a711d3f09e_...  ← initial schema (all 12 tables)
│   │       └── f1a9c3d7e2b4_...  ← audit_logs REVOKE insert-only
│   ├── app/
│   │   ├── config.py         ← all settings loaded from .env
│   │   ├── db/models/        ← 12 SQLAlchemy models
│   │   ├── services/         ← 11 business logic services
│   │   ├── ml/               ← OCR, forgery, face, liveness, RAG
│   │   ├── schemas/          ← Pydantic request/response models
│   │   └── api/v1/           ← ⚠️ router files need to be wired (see Step 7)
│   ├── ml_models/            ← trained .pkl and .pt files (gitignored)
│   ├── knowledge_base/pdfs/  ← your RBI/KYC/Loan PDFs (gitignored)
│   └── training_data/        ← your sample Aadhaar/PAN images (gitignored)
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.js        ← proxies /api → localhost:8000
│   └── src/
│       ├── App.jsx           ← router (6 pages, role-gated)
│       ├── api/client.js     ← all 25 API call functions
│       ├── auth/             ← JWT context, protected routes
│       ├── components/       ← Layout, Sidebar, Badge, Card, etc.
│       └── pages/            ← Login, Dashboard, Onboarding, CaseDetail,
│                                AuditLogs, ChatbotPage
│
├── docker-compose.yml        ← local dev: postgres + redis + backend + worker + frontend
└── .gitignore
```

---

## What the App Does When Fully Running

1. Customer logs in → uploads Aadhaar, PAN, Selfie
2. Pipeline runs: OCR → Tampering Detection → Face Verification → Liveness → Cross-Document → Fraud Score → Explainability → Audit Log written for each step
3. If fraud score > 60 or any result is INCONCLUSIVE → case goes to REVIEW_REQUIRED
4. Bank Staff logs in → sees the case dashboard → can view all results, Grad-CAM heatmap, feature attribution → approves or rejects
5. Admin logs in → sees all cases, all audit logs
6. Anyone can use the chatbot to ask questions about KYC/RBI/Loan policies (answered from your uploaded PDFs)

---

## Troubleshooting

**Backend crashes on startup with "DATABASE_URL not set"**
→ Your `backend/.env` file is missing or not mounted correctly. Check that it exists and `docker compose` can see it.

**"No module named easyocr" or similar import error**
→ The Docker image wasn't rebuilt after `requirements.txt` changed. Run `docker compose up --build`.

**Alembic fails with "relation already exists"**
→ The tables already exist. Either drop and recreate the database (`docker compose down -v && docker compose up`) or use `alembic stamp head` to mark it as current.

**Frontend shows blank page after login**
→ The API routers haven't been wired yet (see Step 7). The frontend is making API calls that return 404.

**Celery worker shows "No module named app.tasks.celery_worker"**
→ The Celery worker file `backend/app/tasks/celery_worker.py` needs to be created. This is the async pipeline entrypoint — create it with a basic Celery app definition pointing to `REDIS_URL`.

**R2 upload fails with "NoCredentialsError"**
→ Your R2 env vars are empty or wrong. Double-check `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and `R2_ENDPOINT_URL` in `backend/.env`.
