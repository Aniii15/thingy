import io
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
