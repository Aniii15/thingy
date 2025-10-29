from pathlib import Path
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
