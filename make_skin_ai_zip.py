import argparse
import shutil
from pathlib import Path

FILES = {
    # Root files
    ".gitignore": """# Python
.venv/
__pycache__/
*.pyc

# Node
node_modules/
dist/

# Models/artifacts
backend/models/model.pth
backend/models/*.pt
backend/models/*.pth
*.log
""",
    "README.md": """# Skin Condition Predictor (Research Only)

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
  - .venv\\Scripts\\Activate.ps1
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
""",

    # Backend
    "backend/requirements.txt": """fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9
pillow==10.4.0
torch>=2.2.0
torchvision>=0.17.0
""",
    "backend/app/__init__.py": "",
    "backend/app/model.py": """import io
import json
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
WEIGHTS_PATH = MODELS_DIR / "model.pth"
LABELS_PATH = MODELS_DIR / "labels.json"

_device = torch.device("cpu")
_model = None
_labels = None
_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def _load_labels() -> Dict[int, str]:
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return {int(k): v for k, v in labels.items()}

def _build_model(num_classes: int):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model

def initialize():
    global _model, _labels
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}. Place your trained 'model.pth' there.")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels file not found at {LABELS_PATH}.")
    _labels = _load_labels()
    num_classes = len(_labels)
    model = _build_model(num_classes)
    state = torch.load(WEIGHTS_PATH, map_location=_device)
    model.load_state_dict(state)
    model.eval()
    _model = model.to(_device)

def predict(image_bytes: bytes, top_k: int = 3) -> List[Dict]:
    global _model, _labels
    if _model is None:
        initialize()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = probs.argsort()[::-1][:top_k]
    results = []
    for idx in top_idx:
        results.append({
            "label": _labels[idx],
            "index": int(idx),
            "probability": float(probs[idx])
        })
    return results
""",
    "backend/app/main.py": """import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from . import model as skin_model

API_KEY = os.getenv("API_KEY")  # optional

app = FastAPI(
    title="Skin Condition Predictor (Research-Only)",
    version="0.1.0",
    description="Uploads patient metadata + lesion image and returns top predicted skin conditions. Not a medical device."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _load():
    try:
        skin_model.initialize()
    except Exception as e:
        print(f"Startup warning: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    blood_pressure: str = Form(...),
    symptoms: str = Form(""),
    medications: str = Form(""),
    api_key: str = Form(default="")
):
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if age < 0 or age > 120:
        raise HTTPException(status_code=400, detail="Invalid age")
    if image.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    try:
        img_bytes = await image.read()
        results = skin_model.predict(img_bytes, top_k=3)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    top = results[0] if results else None

    return JSONResponse({
        "prediction": top,
        "top_k": results,
        "disclaimer": (
            "This tool is for research and educational purposes only, not for diagnosis or treatment. "
            "Always consult a qualified clinician."
        ),
        "echo": {
            "name": name,
            "age": age,
            "blood_pressure": blood_pressure,
            "symptoms": symptoms,
            "medications": medications
        }
    })
""",
    "backend/app/make_dummy_model.py": """from pathlib import Path
import json
import torch
from torchvision import models

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

labels = [
    "acne", "eczema", "psoriasis", "rosacea", "melanoma",
    "benign_keratosis", "basal_cell_carcinoma", "squamous_cell_carcinoma",
    "nevus", "tinea"
]
idx2label = {i: name for i, name in enumerate(labels)}
with open(MODELS_DIR / "labels.json", "w") as f:
    json.dump(idx2label, f)

model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, len(labels))
torch.save(model.state_dict(), MODELS_DIR / "model.pth")

print(f"Wrote {MODELS_DIR / 'model.pth'} and {MODELS_DIR / 'labels.json'} (dummy model).")
print("This is only for smoke testing — predictions are random.")
""",
    "backend/models/.gitkeep": "",

    # Training
    "train/train.py": """import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def get_loaders(data_dir: Path, batch_size=32):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_ds

def build_model(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total

def train(data_dir: Path, out_dir: Path, epochs=10, lr=3e-4, batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, train_ds = get_loaders(data_dir, batch_size=batch_size)
    num_classes = len(train_ds.classes)
    assert num_classes == 10, f"Expected 10 classes, found {num_classes}"

    idx_to_label = {int(idx): label for label, idx in train_ds.class_to_idx.items()}
    with open(out_dir / "labels.json", "w") as f:
        json.dump(idx_to_label, f)

    model = build_model(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_path = out_dir / "model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
        train_loss = running / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  Saved new best to {best_path}")

    print(f"Best val acc: {best_acc:.3f}. Weights saved to {best_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("train/data"))
    p.add_argument("--out_dir", type=Path, default=Path("backend/models"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    train(args.data_dir, args.out_dir, args.epochs, args.lr, args.batch_size)
""",

    # Frontend
    "frontend/package.json": """{
  "name": "skin-ai-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview --port 5173"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "typescript": "^5.6.2",
    "vite": "^5.4.8",
    "@types/react": "^18.3.8",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react-swc": "^3.5.0"
  }
}
""",
    "frontend/tsconfig.json": """{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "Bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true
  },
  "include": ["src"]
}
""",
    "frontend/vite.config.ts": """import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()]
});
""",
    "frontend/index.html": """<!doctype html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Skin AI (Research Only)</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
""",
    "frontend/src/main.tsx": """import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

const root = createRoot(document.getElementById("root")!);
root.render(<App />);
""",
    "frontend/src/App.tsx": """import React, { useState } from "react";

type Prediction = {
  label: string;
  index: number;
  probability: number;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export default function App() {
  const [name, setName] = useState("");
  const [age, setAge] = useState<number | "">("");
  const [bp, setBp] = useState("");
  const [symptoms, setSymptoms] = useState("");
  const [medications, setMeds] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ prediction: Prediction; top_k: Prediction[]; disclaimer: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState("");

  const onFile = (f: File | null) => {
    setImage(f);
    setResult(null);
    setError(null);
    if (f) {
      const url = URL.createObjectURL(f);
      setPreview(url);
    } else {
      setPreview(null);
    }
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    if (!image) {
      setError("Please select an image.");
      return;
    }
    if (!name || age === "" || !bp) {
      setError("Please fill in name, age, and blood pressure.");
      return;
    }
    const fd = new FormData();
    fd.append("image", image);
    fd.append("name", name);
    fd.append("age", String(age));
    fd.append("blood_pressure", bp);
    fd.append("symptoms", symptoms);
    fd.append("medications", medications);
    fd.append("api_key", apiKey);

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: fd
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`${res.status} ${txt}`);
      }
      const json = await res.json();
      setResult(json);
    } catch (err: any) {
      setError(err.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: "20px auto", fontFamily: "system-ui, Arial" }}>
      <h1>Skin Condition Predictor</h1>
      <p style={{ color: "#b45309", background: "#fff7ed", padding: "8px 12px", borderRadius: 8 }}>
        Research/Education only. Not for diagnosis or treatment. Always consult a clinician.
      </p>

      <form onSubmit={submit} style={{ display: "grid", gap: 12, marginTop: 12 }}>
        <div style={{ display: "grid", gap: 8, gridTemplateColumns: "1fr 1fr" }}>
          <label>
            Patient name
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Jane Doe" required
              style={{ width: "100%", padding: 8, marginTop: 4 }} />
          </label>
          <label>
            Age
            <input type="number" value={age} onChange={(e) => setAge(e.target.value ? Number(e.target.value) : "")}
              placeholder="42" required style={{ width: "100%", padding: 8, marginTop: 4 }} />
          </label>
          <label>
            Blood pressure
            <input value={bp} onChange={(e) => setBp(e.target.value)} placeholder="120/80 mmHg" required
              style={{ width: "100%", padding: 8, marginTop: 4 }} />
          </label>
          <label>
            Current medications
            <input value={medications} onChange={(e) => setMeds(e.target.value)} placeholder="(optional)"
              style={{ width: "100%", padding: 8, marginTop: 4 }} />
          </label>
        </div>
        <label>
          Symptoms / notes
          <textarea value={symptoms} onChange={(e) => setSymptoms(e.target.value)} placeholder="Itching, redness..."
            rows={3} style={{ width: "100%", padding: 8, marginTop: 4 }} />
        </label>
        <label>
          Lesion photo (jpg/png)
          <input type="file" accept="image/*" onChange={(e) => onFile(e.target.files?.[0] ?? null)}
            style={{ display: "block", marginTop: 4 }} />
        </label>

        {preview && (
          <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
            <img src={preview} alt="preview" style={{ maxWidth: 280, borderRadius: 8, border: "1px solid #ddd" }} />
            <div style={{ fontSize: 12, color: "#666" }}>
              Make sure the image is in focus and well-lit. Crop to the affected area if possible.
            </div>
          </div>
        )}

        <details>
          <summary>Advanced</summary>
          <label>
            API key (if configured on server)
            <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="optional"
              style={{ width: "100%", padding: 8, marginTop: 4 }} />
          </label>
        </details>

        <button type="submit" disabled={loading} style={{ padding: "10px 14px" }}>
          {loading ? "Analyzing..." : "Predict"}
        </button>
      </form>

      {error && <p style={{ color: "#b91c1c" }}>Error: {error}</p>}

      {result && (
        <div style={{ marginTop: 20 }}>
          <h2>Results</h2>
          <p><b>Top prediction:</b> {result.prediction?.label} ({(result.prediction?.probability * 100).toFixed(1)}%)</p>
          <div style={{ maxWidth: 600 }}>
            {result.top_k.map((p: Prediction) => (
              <div key={p.index} style={{ marginBottom: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>{p.label}</span>
                  <span>{(p.probability * 100).toFixed(1)}%</span>
                </div>
                <div style={{ height: 8, background: "#eee", borderRadius: 4 }}>
                  <div style={{ width: `${p.probability * 100}%`, height: 8, background: "#16a34a", borderRadius: 4 }} />
                </div>
              </div>
            ))}
          </div>
          <p style={{ color: "#555", marginTop: 12 }}>{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}
"""
}


def write_files(dest: Path):
    for rel, content in FILES.items():
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(content)


def main():
    ap = argparse.ArgumentParser(
        description="Generate the Skin-AI project and zip it.")
    ap.add_argument("--out", default="skin-ai",
                    help="Output folder name (default: skin-ai)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing folder if it exists")
    args = ap.parse_args()

    dest = Path(args.out).resolve()
    if dest.exists() and not args.force:
        print(
            f"Destination {dest} already exists. Use --force to overwrite, or choose a different --out name.")
        return

    if dest.exists() and args.force:
        shutil.rmtree(dest)

    print(f"Creating project at {dest} ...")
    write_files(dest)

    zip_base = str(dest)
    archive_path = shutil.make_archive(zip_base, "zip", root_dir=dest)
    print(f"Done. Project folder: {dest}")
    print(f"Zip archive: {archive_path}")

    print("\nNext steps:")
    print(f"  1) cd {dest}")
    print("  2) Create venv and install backend deps (see README.md)")
    print("  3) Generate dummy model (py backend/app/make_dummy_model.py)")
    print("  4) Run backend (uvicorn app.main:app --reload --port 8000)")
    print("  5) In another terminal: cd frontend && npm install && npm run dev")


if __name__ == "__main__":
    main()
