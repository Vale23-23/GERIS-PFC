"""
evaluate.py — Evaluate a trained model and visualize predictions.

Usage:
  python evaluate.py
  python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth
"""

import argparse
import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from dataset import build_datasets
from torch.utils.data import DataLoader


def get_device(cfg):
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    tp = fp = fn = tn = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = (torch.sigmoid(model(images)) > threshold).float()
        tp += (preds * masks).sum().item()
        fp += (preds * (1 - masks)).sum().item()
        fn += ((1 - preds) * masks).sum().item()
        tn += ((1 - preds) * (1 - masks)).sum().item()

    iou       = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n📊 MÉTRICAS DE EVALUACIÓN")
    print(f"{'='*35}")
    print(f"  IoU       : {iou:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


@torch.no_grad()
def visualize_predictions(model, dataset, device, n=4, threshold=0.5):
    model.eval()
    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle("Predicciones del modelo", fontsize=14)

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))
        pred = (pred > threshold).float().squeeze().cpu().numpy()

        rad  = image[0].numpy()
        mask = mask.squeeze().numpy()

        axes[row, 0].imshow(np.log1p(np.abs(rad)), cmap="magma")
        axes[row, 0].set_title("Banda 7 (input)")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask, cmap="Reds", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Máscara real ({int(mask.sum())} px fuego)")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred, cmap="Reds", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Predicción ({int(pred.sum())} px fuego)")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()


def main(cfg_path="config.yaml", checkpoint=None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device     = get_device(cfg)
    checkpoint = checkpoint or cfg["output"]["best_model"]

    if not os.path.exists(checkpoint):
        print(f"❌ No se encontró el checkpoint: {checkpoint}")
        print("   Primero corré train.py para entrenar el modelo.")
        return

    _, val_ds = build_datasets(cfg)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=0)

    n_channels = len(cfg["data"]["bands"])
    model = smp.Unet(
        encoder_name    = cfg["model"]["encoder"],
        encoder_weights = None,
        in_channels     = n_channels,
        classes         = 1,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"✅ Modelo cargado desde: {checkpoint}")

    evaluate(model, val_loader, device)
    visualize_predictions(model, val_ds, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
