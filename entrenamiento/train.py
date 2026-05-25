"""
train.py — Loop de entrenamiento para segmentación de incendios con GOES-19.

Flujo general:
  1. Cargar configuración desde config.yaml
  2. Construir datasets de train y validación
  3. Instanciar el modelo U-Net con encoder ResNet34 preentrenado
  4. Por cada época:
       a. Entrenar sobre el set de train (ajusta los pesos del modelo)
       b. Validar sobre el set de val (solo mide, no ajusta)
       c. Si el modelo mejoró, guardar checkpoint
  5. Al final, reportar el mejor IoU alcanzado

Uso:
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
    """
    Detecta el mejor dispositivo disponible para entrenar:
      - MPS: GPU de Apple Silicon (Mac M1/M2/M3) — más rápido que CPU en Mac
      - CUDA: GPU NVIDIA — el más rápido si está disponible
      - CPU: fallback universal

    Se puede forzar un dispositivo específico en config.yaml con device: cpu/cuda/mps.
    """
    pref = cfg["training"].get("device", "auto")
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(pref)


def dice_loss(pred, target, smooth=1.0):
    """
    Dice Loss: mide el solapamiento entre la predicción y la máscara real.

    Fórmula: 1 - (2 * intersección + smooth) / (suma_pred + suma_target + smooth)

    El parámetro smooth evita división por cero cuando no hay píxeles de fuego.
    Es naturalmente robusto al desbalance de clases porque trabaja con proporciones,
    no con conteos absolutos de píxeles.
    """
    pred   = torch.sigmoid(pred)   # convertir logits a probabilidades [0, 1]
    pred   = pred.view(-1)         # aplanar a vector 1D
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred, target):
    """
    Pérdida combinada: Dice + BCE con peso positivo.

    Se combinan dos pérdidas para atacar el problema de desbalance de clases
    (los píxeles de fuego son una fracción muy pequeña del total):

      - BCE con pos_weight=10: penaliza 10x más los falsos negativos
        (píxeles de fuego que el modelo no detectó). Sin este peso, el modelo
        podría aprender a predecir "todo es no-fuego" y aun así tener pérdida baja.

      - Dice Loss: complementa la BCE midiendo directamente el solapamiento.
        Funciona bien cuando hay muy pocos píxeles positivos.

    La pérdida final es la suma de ambas.
    """
    bce  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([10.0]).to(pred.device)  # penalizar más los falsos negativos
    )(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def iou_score(pred, target, threshold=0.5):
    """
    IoU (Intersection over Union): métrica principal de evaluación.

    Mide qué fracción del área total (unión) está correctamente predicha (intersección).
    Rango: 0 (sin solapamiento) a 1 (predicción perfecta).

    Se aplica un umbral de 0.5 para convertir probabilidades en predicción binaria:
    probabilidad > 0.5 → fuego, probabilidad <= 0.5 → no fuego.
    """
    pred  = (torch.sigmoid(pred) > threshold).float()  # binarizar predicción
    inter = (pred * target).sum()                       # píxeles correctamente predichos como fuego
    union = pred.sum() + target.sum() - inter           # área total cubierta por pred o target
    return (inter + 1e-6) / (union + 1e-6)             # +1e-6 para evitar división por cero


def train_one_epoch(model, loader, optimizer, device):
    """
    Entrena el modelo por una época completa (un pase sobre todo el set de train).

    En cada batch:
      1. Mover datos al dispositivo (GPU/CPU)
      2. Forward pass: el modelo predice las máscaras
      3. Calcular la pérdida comparando predicción vs máscara real
      4. Backward pass: calcular gradientes
      5. Optimizer step: actualizar los pesos del modelo

    Retorna: (pérdida promedio, IoU promedio) sobre todos los batches.
    """
    model.train()  # activar modo entrenamiento (habilita dropout, batch norm, etc.)
    total_loss, total_iou = 0.0, 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()          # limpiar gradientes del paso anterior
        preds = model(images)          # forward pass: predecir máscaras
        loss  = combined_loss(preds, masks)  # calcular pérdida
        loss.backward()                # backward pass: calcular gradientes
        optimizer.step()               # actualizar pesos

        total_loss += loss.item()
        total_iou  += iou_score(preds, masks).item()

    n = len(loader)  # cantidad de batches
    return total_loss / n, total_iou / n


@torch.no_grad()  # desactivar cálculo de gradientes para ahorrar memoria
def validate(model, loader, device):
    """
    Evalúa el modelo sobre el set de validación sin actualizar los pesos.

    @torch.no_grad() desactiva el cálculo de gradientes, lo que reduce el uso
    de memoria y acelera la evaluación (no necesitamos backprop aquí).

    Retorna: (pérdida promedio, IoU promedio) sobre el set de validación.
    """
    model.eval()  # activar modo evaluación (desactiva dropout, fija batch norm)
    total_loss, total_iou = 0.0, 0.0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        total_loss += combined_loss(preds, masks).item()
        total_iou  += iou_score(preds, masks).item()

    n = len(loader)
    return total_loss / n, total_iou / n


def main(cfg_path="config.yaml", limit=None):
    # ── 1. Cargar configuración ──────────────────────────────────────────────
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg)
    print(f"🖥  Usando dispositivo: {device}")

    # ── 2. Construir datasets y dataloaders ──────────────────────────────────
    # build_datasets descubre los timestamps válidos, divide train/val y
    # normaliza usando las stats del set de entrenamiento.
    train_ds, val_ds = build_datasets(cfg, limit=limit)

    # DataLoader se encarga de armar los batches y mezclar los datos en cada época
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,    # mezclar en cada época para evitar que el modelo memorice el orden
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,   # no mezclar validación (no importa el orden)
        num_workers=0,
    )

    # ── 3. Instanciar el modelo ──────────────────────────────────────────────
    # U-Net con encoder ResNet34 preentrenado en ImageNet.
    # El encoder extrae características de la imagen a distintas escalas.
    # El decoder reconstruye la máscara píxel a píxel usando esas características.
    # encoder_weights="imagenet" carga pesos preentrenados → transfer learning.
    n_channels = len(cfg["data"]["bands"])  # cantidad de bandas satelitales como entrada
    model = smp.Unet(
        encoder_name    = cfg["model"]["encoder"],          # "resnet34"
        encoder_weights = cfg["model"]["encoder_weights"],  # "imagenet"
        in_channels     = n_channels,
        classes         = 1,   # salida binaria: fuego / no fuego
    ).to(device)

    # ── 4. Optimizador y scheduler ───────────────────────────────────────────
    # Adam: optimizador adaptativo, buen punto de partida para la mayoría de los modelos
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    # ReduceLROnPlateau: si la pérdida de validación no mejora en 5 épocas,
    # reduce el learning rate a la mitad. Permite afinar el modelo cuando
    # ya está cerca de una buena solución.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # ── 5. Loop de entrenamiento ─────────────────────────────────────────────
    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    best_val_iou = 0.0  # rastrear el mejor IoU de validación para guardar el mejor modelo
    epochs       = cfg["training"]["epochs"]

    print(f"\n🚀 Iniciando entrenamiento — {epochs} épocas\n")

    for epoch in range(1, epochs + 1):
        # Fase de entrenamiento: el modelo aprende ajustando sus pesos
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device)

        # Fase de validación: medimos rendimiento en datos no vistos (sin actualizar pesos)
        val_loss, val_iou = validate(model, val_loader, device)

        # Ajustar learning rate si la pérdida de validación se estancó
        scheduler.step(val_loss)

        print(f"Época {epoch:3d}/{epochs}  "
              f"train loss: {train_loss:.4f}  train IoU: {train_iou:.4f}  |  "
              f"val loss: {val_loss:.4f}  val IoU: {val_iou:.4f}")

        # Guardar el modelo solo si mejoró en validación (early saving)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), cfg["output"]["best_model"])
            print(f"  ✅ Mejor modelo guardado (val IoU: {best_val_iou:.4f})")

    print(f"\n✔ Entrenamiento completo. Mejor val IoU: {best_val_iou:.4f}")
    print(f"  Modelo guardado en: {cfg['output']['best_model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Cantidad máxima de timestamps a usar")
    args = parser.parse_args()
    main(args.config, limit=args.limit)
