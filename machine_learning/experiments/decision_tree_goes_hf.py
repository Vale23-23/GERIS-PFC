import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.tracking import create_tracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple GOES pixel-wise decision tree trained on a real scene from the Hugging Face cached dataset.")
    parser.add_argument("--date", default="20251115_0850", help="GOES scene timestamp, e.g. 20251115_0850")
    parser.add_argument("--dataset-root", default="implementacion/dataset/uruguay", help="Folder containing the GOES bands and fire mask.")
    parser.add_argument("--experiment-name", default=os.getenv("COMET_EXPERIMENT_NAME"), help="Comet experiment name.")
    parser.add_argument("--tags", default=os.getenv("COMET_TAGS", "goes,hf,tree"), help="Comma-separated Comet tags.")
    parser.add_argument("--max-samples-per-class", type=int, default=2000, help="Maximum number of pixels to sample per class.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction reserved for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum depth of the tree.")
    parser.add_argument("--min-samples-leaf", type=int, default=20, help="Minimum number of samples per leaf.")
    parser.add_argument("--no-comet", action="store_true", help="Disable Comet logging for local-only runs.")
    return parser.parse_args()


def load_scene(dataset_root: str, date_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(dataset_root).expanduser().resolve()
    b07 = np.load(root / "ABI-L1b-Rad-B07" / f"{date_str}.npy")
    b14 = np.load(root / "ABI-L1b-Rad-B14" / f"{date_str}.npy")
    fire = np.load(root / "ABI-L2-FDCF-Mask" / f"{date_str}.npy")
    return b07.astype(np.float32), b14.astype(np.float32), fire.astype(np.int16)


def build_training_set(
    b07: np.ndarray,
    b14: np.ndarray,
    fire: np.ndarray,
    max_samples_per_class: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = np.isfinite(b07) & np.isfinite(b14) & np.isfinite(fire)
    b07 = b07[valid_mask]
    b14 = b14[valid_mask]
    fire = fire[valid_mask]

    y = (fire > 0).astype(int)
    X = np.column_stack([
        b07.ravel(),
        b14.ravel(),
        (b07 - b14).ravel(),
    ])

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng = np.random.default_rng(random_state)
    if len(pos_idx) > max_samples_per_class:
        pos_idx = rng.choice(pos_idx, size=max_samples_per_class, replace=False)
    if len(neg_idx) > max_samples_per_class:
        neg_idx = rng.choice(neg_idx, size=max_samples_per_class, replace=False)

    idx = np.concatenate([pos_idx, neg_idx])
    X = X[idx]
    y = y[idx]
    return X, y


def main() -> None:
    args = parse_args()

    if args.no_comet:
        os.environ["COMET_ENABLED"] = "false"

    b07, b14, fire = load_scene(args.dataset_root, args.date)
    X, y = build_training_set(b07, b14, fire, args.max_samples_per_class, args.random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        random_state=args.random_state,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
    )

    experiment_name = args.experiment_name or os.getenv("COMET_EXPERIMENT_NAME") or (
        f"goes-tree-{args.date}"
    )
    os.environ["COMET_EXPERIMENT_NAME"] = experiment_name
    if args.tags:
        os.environ["COMET_TAGS"] = args.tags

    tracker = create_tracker(experiment_name=experiment_name)
    tracker.log_parameters(
        {
            "model": "DecisionTreeClassifier",
            "dataset_date": args.date,
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "feature_columns": "b07,b14,b07_minus_b14",
            "target_definition": "fire_mask > 0",
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_samples_per_class": int(args.max_samples_per_class),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "positive_rate": float(y.mean()),
        }
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    tracker.log_metrics(metrics, step=1)
    tracker.end()

    print(f"Scene date: {args.date}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
