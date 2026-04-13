"""
dataset.py — Dataset de PyTorch para detección de incendios con GOES-19.

Carga la Banda 7 de radiancia (.npy) y la máscara de fuego FDCF (.npy) para
cada timestamp. Devuelve el tensor de entrada normalizado y la máscara binaria
de fuego lista para entrenar.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class GOESFireDataset(Dataset):
    def __init__(self, timestamps, dataset_root, bands, mask_product, image_size=256, stats=None):
        """
        timestamps:   lista de strings con timestamps, ej. ["20250901_1100", ...]
        dataset_root: ruta a dataset/uruguay/
        bands:        lista de carpetas de bandas, ej. ["ABI-L1b-Rad-B07"]
        mask_product: carpeta de la máscara de fuego, ej. "ABI-L2-FDCF"
        image_size:   tamaño de recorte (H, W) — todas las imágenes se llevan a este tamaño
        stats:        stats de normalización pre-calculadas (del set de entrenamiento).
                      Si es None, se calculan sobre los timestamps de este dataset.
                      Siempre pasar train_ds.stats al dataset de validación/test
                      para evitar data leakage.
        """
        self.timestamps   = timestamps
        self.dataset_root = dataset_root
        self.bands        = bands
        self.mask_product = mask_product
        self.image_size   = image_size

        # Si no se pasan stats externas, se calculan sobre este dataset.
        # Para validación siempre se deben pasar las stats del set de entrenamiento.
        self.stats = stats if stats is not None else self._compute_stats()

    def _compute_stats(self):
        """
        Calcula media y desviación estándar por banda sobre todos los timestamps.
        Se usa para normalizar las imágenes antes de pasarlas al modelo.
        Normalizar es importante para que el modelo no dependa de las unidades
        físicas del sensor (radiancia) y para estabilizar el entrenamiento.
        """
        stats = {}
        for band in self.bands:
            values = []
            for ts in self.timestamps:
                path = os.path.join(self.dataset_root, band, f"{ts}.npy")
                if os.path.exists(path):
                    arr = np.load(path).astype(np.float32)
                    values.append(arr.flatten())  # aplanamos para calcular stats globales
            if values:
                all_vals = np.concatenate(values)
                stats[band] = {
                    "mean": float(all_vals.mean()),
                    "std":  float(all_vals.std()) + 1e-6  # +1e-6 para evitar división por cero
                }
            else:
                # Fallback si no hay archivos: normalización identidad
                stats[band] = {"mean": 0.0, "std": 1.0}
        return stats

    def _load_and_normalize(self, ts):
        """
        Carga todas las bandas para un timestamp y las normaliza con z-score:
            x_norm = (x - media) / std
        Devuelve un array de forma (C, H, W) donde C = número de bandas.
        """
        channels = []
        for band in self.bands:
            path = os.path.join(self.dataset_root, band, f"{ts}.npy")
            arr  = np.load(path).astype(np.float32)
            # Normalización z-score usando las stats del set de entrenamiento
            arr  = (arr - self.stats[band]["mean"]) / self.stats[band]["std"]
            channels.append(arr)
        # Apilamos las bandas en el eje 0 → (C, H, W)
        return np.stack(channels, axis=0)

    def _load_mask(self, ts):
        """
        Carga la máscara de fuego del producto ABI-L2-FDCF.
        El campo DQF (Data Quality Flag) indica la confianza de detección:
            DQF = 0 → fuego detectado con alta confianza → marcamos como 1
            cualquier otro valor → sin fuego o baja confianza → marcamos como 0
        Devuelve un array binario (H, W): 1 = fuego, 0 = no fuego.
        """
        path = os.path.join(self.dataset_root, self.mask_product, f"{ts}.npy")
        mask = np.load(path).astype(np.int8)
        return (mask == 0).astype(np.float32)  # DQF=0 → fuego confirmado

    def _crop_or_pad(self, arr, target):
        """
        Lleva el array al tamaño target×target mediante:
          1. Padding con reflexión si la imagen es más chica que target
          2. Recorte central si la imagen es más grande que target

        Esto es necesario porque PyTorch requiere que todas las imágenes
        de un batch tengan exactamente el mismo tamaño.
        """
        h, w = arr.shape[-2], arr.shape[-1]

        # Paso 1: padding si la imagen es más chica que el target
        pad_h = max(0, target - h)
        pad_w = max(0, target - w)
        if arr.ndim == 3:  # imagen con canales (C, H, W)
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        else:              # máscara sin canal (H, W)
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")

        # Paso 2: recorte central si la imagen es más grande que el target
        h, w = arr.shape[-2], arr.shape[-1]
        y = (h - target) // 2  # offset vertical para centrar
        x = (w - target) // 2  # offset horizontal para centrar
        if arr.ndim == 3:
            return arr[:, y:y+target, x:x+target]
        return arr[y:y+target, x:x+target]

    def __len__(self):
        # Cantidad total de muestras en el dataset
        return len(self.timestamps)

    def __getitem__(self, idx):
        """
        Devuelve una muestra del dataset: (imagen, máscara).
        PyTorch llama a este método automáticamente al iterar el DataLoader.

        Retorna:
            image: tensor (C, H, W) — bandas satelitales normalizadas
            mask:  tensor (1, H, W) — máscara binaria de fuego
        """
        ts = self.timestamps[idx]

        # Cargar y normalizar la imagen satelital
        image = self._load_and_normalize(ts)   # (C, H, W)

        # Cargar la máscara de fuego (ground truth)
        mask  = self._load_mask(ts)            # (H, W)

        # Llevar ambos al tamaño fijo requerido por el modelo
        image = self._crop_or_pad(image, self.image_size)
        mask  = self._crop_or_pad(mask,  self.image_size)

        return (
            torch.tensor(image, dtype=torch.float32),
            # unsqueeze(0) agrega el eje de canal → (1, H, W), requerido por la U-Net
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
        )


def build_datasets(cfg):
    """
    Construye los datasets de entrenamiento y validación a partir del config.

    Flujo:
      1. Descubre todos los timestamps que tienen TODOS los archivos requeridos
      2. Los mezcla aleatoriamente y los divide en train/val según val_split
      3. Calcula las stats de normalización SOLO sobre el set de entrenamiento
      4. Pasa esas stats al dataset de validación para evitar data leakage

    Retorna: (train_dataset, val_dataset)
    """
    import random

    root         = cfg["data"]["dataset_root"]
    bands        = cfg["data"]["bands"]
    mask_product = cfg["data"]["mask_product"]
    image_size   = cfg["data"]["image_size"]
    val_split    = cfg["data"]["val_split"]

    # Descubrir timestamps válidos: solo los que tienen TODOS los productos descargados.
    # Si falta la banda o la máscara para una hora, ese timestamp se descarta.
    mask_folder = os.path.join(root, mask_product)
    all_ts = sorted([
        f.replace(".npy", "")
        for f in os.listdir(mask_folder)
        if f.endswith(".npy")
        and all(os.path.exists(os.path.join(root, b, f)) for b in bands)
    ])

    if not all_ts:
        raise RuntimeError(f"No se encontraron timestamps completos en {root}")

    # Mezcla aleatoria antes de dividir para evitar sesgos temporales
    random.shuffle(all_ts)

    # División train/val: ej. val_split=0.2 → 80% train, 20% val
    split    = int(len(all_ts) * (1 - val_split))
    train_ts = all_ts[:split]
    val_ts   = all_ts[split:]

    print(f"📊 Dataset: {len(all_ts)} timestamps  |  train: {len(train_ts)}  |  val: {len(val_ts)}")

    # Crear dataset de entrenamiento (calcula sus propias stats)
    train_ds = GOESFireDataset(train_ts, root, bands, mask_product, image_size)

    # Crear dataset de validación usando las stats del entrenamiento.
    # Esto es correcto: en producción tampoco conocemos las stats del dato nuevo,
    # así que siempre normalizamos con lo que aprendimos del set de entrenamiento.
    val_ds = GOESFireDataset(val_ts, root, bands, mask_product, image_size, stats=train_ds.stats)

    return train_ds, val_ds
