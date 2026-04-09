"""
train.py — Training loop for GOES-19 fire segmentation.

Usage:
  python train.py
  python train.py --config config.yaml
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import build_datasets


def get_device(cfg):
    pref = cfg["training"].get("device", "auto")
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(pref)


def dice_loss(pred, target, smooth=1.0):
    pred   = torch.sigmoid(pred)
    pred   = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred, target):
    """Dice + BCE — works well for imbalanced segmentation."""
    bce  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(pred.device))(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def iou_score(pred, target, threshold=0.5):
    pred   = (torch.sigmoid(pred) > threshold).float()
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return (inter + 1e-6) / (union + 1e-6)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_iou = 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss  = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_iou  += iou_score(preds, masks).item()
    n = len(loader)
    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, total_iou = 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        total_loss += combined_loss(preds, masks).item()
        total_iou  += iou_score(preds, masks).item()
    n = len(loader)
    return total_loss / n, total_iou / n


def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg)
    print(f"🖥  Usando dispositivo: {device}")

    # Data
    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=0)

    # Model — U-Net with pretrained ResNet34 encoder
    n_channels = len(cfg["data"]["bands"])
    model = smp.Unet(
        encoder_name    = cfg["model"]["encoder"],
        encoder_weights = cfg["model"]["encoder_weights"],
        in_channels     = n_channels,
        classes         = 1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    best_val_iou = 0.0
    epochs       = cfg["training"]["epochs"]

    print(f"\n🚀 Iniciando entrenamiento — {epochs} épocas\n")
    for epoch in range(1, epochs + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device)
        val_loss,   val_iou   = validate(model, val_loader, device)
        scheduler.step(val_loss)

        print(f"Época {epoch:3d}/{epochs}  "
              f"train loss: {train_loss:.4f}  train IoU: {train_iou:.4f}  |  "
              f"val loss: {val_loss:.4f}  val IoU: {val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), cfg["output"]["best_model"])
            print(f"  ✅ Mejor modelo guardado (val IoU: {best_val_iou:.4f})")

    print(f"\n✔ Entrenamiento completo. Mejor val IoU: {best_val_iou:.4f}")
    print(f"  Modelo guardado en: {cfg['output']['best_model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
