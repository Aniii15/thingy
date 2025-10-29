# Skin Condition Predictor (Research Only)

This is a full-stack demo: FastAPI backend + React (Vite) frontend + PyTorch model.
- Upload patient details + a skin lesion image
- Get top-3 predictions from a 10-class classifier

Important
- Research/education only. Not a medical device. Do not use for diagnosis or treatment.
- If you handle real patient data, you must implement consent, encryption, access control, and comply with regulations (HIPAA/GDPR).

## What you need
- Python 3.9–3.12
- Node.js 18+ (includes npm)

## Setup: Backend (FastAPI + PyTorch)
1) Open a terminal in the project root (skin-ai).
2) Create a virtual environment and install deps:

- Windows (PowerShell)
  - py -m venv .venv
  - .venv\Scripts\Activate.ps1
- macOS/Linux
  - python3 -m venv .venv
  - source .venv/bin/activate

Then:
- pip install --upgrade pip
- pip install -r backend/requirements.txt
- If PyTorch install fails, see https://pytorch.org/get-started/locally for the exact command for your OS.

3) Create a dummy model (for a quick smoke test; predictions are random):
- Windows: py backend/app/make_dummy_model.py
- macOS/Linux: python3 backend/app/make_dummy_model.py
This writes:
- backend/models/model.pth
- backend/models/labels.json

4) Start the backend:
- cd backend
- uvicorn app.main:app --reload --port 8000
- Test health check at http://127.0.0.1:8000/health

Optional: Train a real model (longer, for real predictions)
- Prepare dataset:
  - train/data/train/<10 class folders>/*.jpg
  - train/data/val/<10 class folders>/*.jpg
- From project root (venv active):
  - pip install -r backend/requirements.txt
  - pip install torch torchvision
  - python train/train.py --data_dir train/data --out_dir backend/models --epochs 12

## Setup: Frontend (React + Vite)
1) Open a second terminal in the project root (keep the backend running).
2) Install and start:
- cd frontend
- npm install
- npm run dev
- Open the URL shown (usually http://localhost:5173)

## Using the app
1) Fill in name, age, blood pressure (and optionally symptoms/medications).
2) Upload a JPG/PNG of the lesion.
3) Click Predict.
- With the dummy model, results are random (for testing the flow).
- With a trained model, you’ll get meaningful predictions.

## Troubleshooting
- “Model weights not found”: run the dummy model script or train a model.
- CORS error: ensure frontend runs on http://localhost:5173 or add your port in backend/app/main.py allow_origins.
- 422 errors: make sure required fields and image are provided.
- Pillow/format errors: use .jpg/.jpeg/.png.
- Uvicorn not found: activate the virtual environment and reinstall requirements.

## Security and clinical notes
- Sample app does not persist data. If you store PHI, add consent, encryption, RBAC, audit logging, HTTPS.
- Validate performance across diverse skin tones/demographics, document limitations.
- Not a medical device; clinical use requires formal validation and regulatory clearance.
