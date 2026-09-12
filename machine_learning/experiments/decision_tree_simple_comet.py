import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
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
    parser = argparse.ArgumentParser(description="Small decision-tree experiment with optional Comet tracking.")
    parser.add_argument("--experiment-name", default=os.getenv("COMET_EXPERIMENT_NAME"), help="Name for the Comet run.")
    parser.add_argument("--project-name", default=os.getenv("COMET_PROJECT_NAME", "geris-pfc-ml"), help="Comet project name.")
    parser.add_argument("--workspace", default=os.getenv("COMET_WORKSPACE"), help="Comet workspace name.")
    parser.add_argument("--tags", default=os.getenv("COMET_TAGS", "toy,baseline"), help="Comma-separated Comet tags.")
    parser.add_argument("--comet-enabled", action="store_true", default=os.getenv("COMET_ENABLED", "false").lower() in {"1", "true", "yes", "on"}, help="Enable Comet tracking for this run.")
    return parser.parse_args()


@dataclass
class Config:
    random_state: int = 42
    test_size: float = 0.25
    max_depth: int = 3
    min_samples_leaf: int = 5


def build_toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Tiny synthetic dataset for a simple fire/no-fire binary problem."""
    X = np.array(
        [
            [28.0, 42.0],
            [32.0, 38.0],
            [21.0, 68.0],
            [35.0, 28.0],
            [29.0, 44.0],
            [18.0, 73.0],
            [34.0, 24.0],
            [31.0, 36.0],
            [23.0, 61.0],
            [36.0, 22.0],
            [26.0, 55.0],
            [40.0, 18.0],
            [19.0, 71.0],
            [30.0, 30.0],
            [33.0, 32.0],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1], dtype=int)
    return X, y


def main() -> None:
    args = parse_args()
    cfg = Config()
    X, y = build_toy_dataset()

    if args.project_name:
        os.environ["COMET_PROJECT_NAME"] = args.project_name
    if args.workspace:
        os.environ["COMET_WORKSPACE"] = args.workspace
    if args.tags:
        os.environ["COMET_TAGS"] = args.tags
    if args.experiment_name:
        os.environ["COMET_EXPERIMENT_NAME"] = args.experiment_name
    os.environ["COMET_ENABLED"] = "true" if args.comet_enabled else "false"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        random_state=cfg.random_state,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
    )

    experiment_name = os.getenv("COMET_EXPERIMENT_NAME") or (
        f"toy-decision-tree-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    )
    tracker = create_tracker(experiment_name=experiment_name)
    tracker.log_parameters(
        {
            "model": "DecisionTreeClassifier",
            "experiment_name": experiment_name,
            "random_state": cfg.random_state,
            "test_size": cfg.test_size,
            "max_depth": cfg.max_depth,
            "min_samples_leaf": cfg.min_samples_leaf,
            "dataset_size": int(len(X)),
            "n_features": int(X.shape[1]),
            "target_positive_rate": float(np.mean(y)),
            "dataset_kind": "toy_synthetic_fire_detection",
            "comet_enabled": str(os.getenv("COMET_ENABLED", "false")).lower() in {"1", "true", "yes", "on"},
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

    print("Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
