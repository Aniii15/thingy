import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_loaders(data_dir: Path, batch_size=32):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
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


def train(data_dir: Path, out_dir: Path, epochs=10, lr=3e-4, batch_size=32, device=None):
    device = device or pick_device()
    print(f"Using device: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, train_ds = get_loaders(
        data_dir, batch_size=batch_size)
    num_classes = len(train_ds.classes)
    print(f"Detected classes ({num_classes}): {train_ds.classes}")

    # Save index-to-label mapping for inference
    idx_to_label = {int(idx): label for label,
                    idx in train_ds.class_to_idx.items()}
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
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  Saved new best to {best_path}")

    print(f"Best val acc: {best_acc:.3f}. Weights saved to {best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("train/data"))
    p.add_argument("--out_dir", type=Path, default=Path("backend/models"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default=None,
                   help="cpu|cuda|mps (default: auto)")
    args = p.parse_args()
    train(args.data_dir, args.out_dir, args.epochs,
          args.lr, args.batch_size, args.device)
