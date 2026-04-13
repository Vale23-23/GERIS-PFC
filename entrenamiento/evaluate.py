"""
evaluate.py — Evaluación del modelo entrenado y visualización de predicciones.

Carga el checkpoint guardado por train.py y calcula métricas sobre el set de
validación. También genera visualizaciones comparando la imagen de entrada,
la máscara real y la predicción del modelo.

Uso:
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
    """Detecta el mejor dispositivo disponible (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()  # no necesitamos gradientes para evaluar
def evaluate(model, loader, device, threshold=0.5):
    """
    Calcula métricas de evaluación sobre el set de validación completo.

    Para cada batch, binariza las predicciones con el umbral y acumula:
      - TP (True Positives):  píxeles de fuego correctamente detectados
      - FP (False Positives): píxeles predichos como fuego que no lo son
      - FN (False Negatives): píxeles de fuego que el modelo no detectó
      - TN (True Negatives):  píxeles sin fuego correctamente ignorados

    A partir de esos conteos calcula:
      - IoU:       intersección / unión — métrica principal de segmentación
      - F1 Score:  media armónica de precision y recall
      - Precision: de todo lo que predije como fuego, ¿cuánto era fuego real?
      - Recall:    de todo el fuego real, ¿cuánto detecté?

    Un Recall alto es especialmente importante en detección de incendios:
    es peor no detectar un incendio real (FN) que dar una falsa alarma (FP).
    """
    model.eval()
    tp = fp = fn = tn = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        # Convertir logits a probabilidades y binarizar con el umbral
        preds = (torch.sigmoid(model(images)) > threshold).float()

        # Acumular conteos sobre todos los píxeles del batch
        tp += (preds * masks).sum().item()              # predijo fuego, era fuego
        fp += (preds * (1 - masks)).sum().item()        # predijo fuego, no era fuego
        fn += ((1 - preds) * masks).sum().item()        # no predijo fuego, era fuego
        tn += ((1 - preds) * (1 - masks)).sum().item()  # no predijo fuego, no era fuego

    # Calcular métricas finales con suavizado para evitar división por cero
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
    """
    Genera una grilla de visualizaciones con n muestras aleatorias del dataset.

    Cada fila muestra tres paneles:
      - Izquierda:  Banda 7 (infrarrojo térmico) — la entrada al modelo
      - Centro:     Máscara real (ground truth del producto FDCF)
      - Derecha:    Predicción del modelo

    La Banda 7 se visualiza con escala logarítmica (log1p) para resaltar
    los focos de calor que de otro modo quedarían aplastados por el rango dinámico.
    """
    model.eval()
    # Seleccionar n muestras aleatorias del dataset
    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle("Predicciones del modelo", fontsize=14)

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]

        # Predicción: forward pass con la imagen individual
        pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))  # agregar dim de batch
        pred = (pred > threshold).float().squeeze().cpu().numpy()   # binarizar y llevar a numpy

        rad  = image[0].numpy()       # primera banda (Banda 7) para visualizar
        mask = mask.squeeze().numpy() # quitar dim de canal

        # Panel izquierdo: imagen de entrada (escala log para mejor contraste)
        axes[row, 0].imshow(np.log1p(np.abs(rad)), cmap="magma")
        axes[row, 0].set_title("Banda 7 (input)")
        axes[row, 0].axis("off")

        # Panel central: máscara real del satélite
        axes[row, 1].imshow(mask, cmap="Reds", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Máscara real ({int(mask.sum())} px fuego)")
        axes[row, 1].axis("off")

        # Panel derecho: predicción del modelo
        axes[row, 2].imshow(pred, cmap="Reds", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Predicción ({int(pred.sum())} px fuego)")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()


def main(cfg_path="config.yaml", checkpoint=None):
    # ── 1. Cargar configuración ──────────────────────────────────────────────
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device     = get_device(cfg)
    checkpoint = checkpoint or cfg["output"]["best_model"]

    # Verificar que el checkpoint existe antes de continuar
    if not os.path.exists(checkpoint):
        print(f"❌ No se encontró el checkpoint: {checkpoint}")
        print("   Primero corré train.py para entrenar el modelo.")
        return

    # ── 2. Construir dataset de validación ───────────────────────────────────
    # build_datasets devuelve (train_ds, val_ds); solo usamos val_ds para evaluar
    _, val_ds = build_datasets(cfg)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # ── 3. Cargar el modelo con los pesos del checkpoint ─────────────────────
    # encoder_weights=None porque vamos a cargar los pesos desde el archivo,
    # no desde ImageNet (el modelo ya fue fine-tuneado durante el entrenamiento)
    n_channels = len(cfg["data"]["bands"])
    model = smp.Unet(
        encoder_name    = cfg["model"]["encoder"],
        encoder_weights = None,   # no cargar ImageNet, usamos el checkpoint guardado
        in_channels     = n_channels,
        classes         = 1,
    ).to(device)

    # Cargar los pesos guardados por train.py
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"✅ Modelo cargado desde: {checkpoint}")

    # ── 4. Evaluar y visualizar ──────────────────────────────────────────────
    evaluate(model, val_loader, device)
    visualize_predictions(model, val_ds, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
