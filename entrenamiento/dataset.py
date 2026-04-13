"""
dataset.py — PyTorch Dataset for GOES-19 fire detection.

Loads Band 7 radiance (.npy) and FDCF fire mask (.npy) for each timestamp.
Returns normalized input tensor and binary fire mask.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class GOESFireDataset(Dataset):
    def __init__(self, timestamps, dataset_root, bands, mask_product, image_size=256, stats=None):
        """
        timestamps:   list of timestamp strings, e.g. ["20250901_1100", ...]
        dataset_root: path to dataset/uruguay/
        bands:        list of band folder names, e.g. ["ABI-L1b-Rad-B07"]
        mask_product: folder name for the fire mask, e.g. "ABI-L2-FDCF"
        image_size:   crop size (H, W) — all images resized to this
        stats:        optional pre-computed normalization stats (from training set).
                      If None, stats are computed from this dataset's timestamps.
                      Always pass train_ds.stats to validation/test datasets to avoid data leakage.
        """
        self.timestamps   = timestamps
        self.dataset_root = dataset_root
        self.bands        = bands
        self.mask_product = mask_product
        self.image_size   = image_size

        # Use provided stats (from train set) or compute from this dataset's timestamps
        self.stats = stats if stats is not None else self._compute_stats()

    def _compute_stats(self):
        """Compute mean and std per band across all timestamps."""
        stats = {}
        for band in self.bands:
            values = []
            for ts in self.timestamps:
                path = os.path.join(self.dataset_root, band, f"{ts}.npy")
                if os.path.exists(path):
                    arr = np.load(path).astype(np.float32)
                    values.append(arr.flatten())
            if values:
                all_vals = np.concatenate(values)
                stats[band] = {"mean": float(all_vals.mean()), "std": float(all_vals.std()) + 1e-6}
            else:
                stats[band] = {"mean": 0.0, "std": 1.0}
        return stats

    def _load_and_normalize(self, ts):
        """Load and normalize all bands for a timestamp. Returns (C, H, W) tensor."""
        channels = []
        for band in self.bands:
            path = os.path.join(self.dataset_root, band, f"{ts}.npy")
            arr  = np.load(path).astype(np.float32)
            arr  = (arr - self.stats[band]["mean"]) / self.stats[band]["std"]
            channels.append(arr)
        return np.stack(channels, axis=0)  # (C, H, W)

    def _load_mask(self, ts):
        """Load fire mask. Returns binary (H, W) array: 1=fire, 0=no fire."""
        path = os.path.join(self.dataset_root, self.mask_product, f"{ts}.npy")
        mask = np.load(path).astype(np.int8)
        return (mask == 0).astype(np.float32)  # DQF=0 → high confidence fire

    def _crop_or_pad(self, arr, target):
        """Center-crop or pad array to target size."""
        h, w = arr.shape[-2], arr.shape[-1]
        # Pad if smaller
        pad_h = max(0, target - h)
        pad_w = max(0, target - w)
        if arr.ndim == 3:
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        else:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="reflect")
        # Center crop
        h, w = arr.shape[-2], arr.shape[-1]
        y = (h - target) // 2
        x = (w - target) // 2
        if arr.ndim == 3:
            return arr[:, y:y+target, x:x+target]
        return arr[y:y+target, x:x+target]

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        ts    = self.timestamps[idx]
        image = self._load_and_normalize(ts)           # (C, H, W)
        mask  = self._load_mask(ts)                    # (H, W)

        image = self._crop_or_pad(image, self.image_size)
        mask  = self._crop_or_pad(mask,  self.image_size)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask,  dtype=torch.float32).unsqueeze(0),  # (1, H, W)
        )


def build_datasets(cfg):
    """
    Build train/val datasets from config.
    Returns (train_dataset, val_dataset).
    """
    import random
    root         = cfg["data"]["dataset_root"]
    bands        = cfg["data"]["bands"]
    mask_product = cfg["data"]["mask_product"]
    image_size   = cfg["data"]["image_size"]
    val_split    = cfg["data"]["val_split"]

    # Discover all timestamps that have all required files
    mask_folder = os.path.join(root, mask_product)
    all_ts = sorted([
        f.replace(".npy", "")
        for f in os.listdir(mask_folder)
        if f.endswith(".npy")
        and all(os.path.exists(os.path.join(root, b, f)) for b in bands)
    ])

    if not all_ts:
        raise RuntimeError(f"No se encontraron timestamps completos en {root}")

    random.shuffle(all_ts)
    split     = int(len(all_ts) * (1 - val_split))
    train_ts  = all_ts[:split]
    val_ts    = all_ts[split:]

    print(f"📊 Dataset: {len(all_ts)} timestamps  |  train: {len(train_ts)}  |  val: {len(val_ts)}")

    train_ds = GOESFireDataset(train_ts, root, bands, mask_product, image_size)
    # Pass train stats to val dataset to avoid data leakage
    val_ds   = GOESFireDataset(val_ts, root, bands, mask_product, image_size, stats=train_ds.stats)
    return train_ds, val_ds
