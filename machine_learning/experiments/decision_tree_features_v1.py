"""Train a reproducible Decision Tree on per-scene feature CSVs.

Unlike the legacy decision_tree_v1.py, this runner uses the feature contract
emitted by build_features.py and keeps train/test scenes disjoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.tracking import create_tracker

PROTECTED_COLUMNS = {"timestamp", "i", "j", "ref_code", "ref_fire"}
DEFAULT_FEATURES_DIR = PROJECT_ROOT / "machine_learning" / "data" / "features"
DEFAULT_FEATURE_SETS = PROJECT_ROOT / "machine_learning" / "configs" / "feature_sets.yaml"
DEFAULT_EXPERIMENT_CONFIG = PROJECT_ROOT / "machine_learning" / "configs" / "experiments" / "dt_features_baseline_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decision Tree reproducible sobre CSVs de features por escena.")
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--feature-sets-config", default=None)
    parser.add_argument("--experiment-config", default=str(DEFAULT_EXPERIMENT_CONFIG))
    parser.add_argument("--feature-set", default=None, help="Set YAML o combinación set_a+set_b; también 'all'.")
    parser.add_argument("--feature-columns", nargs="+", default=None, help="Columnas explícitas, alternativa a --feature-set.")
    parser.add_argument("--target", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--split-id", default=None)
    parser.add_argument("--train-dates", default=None, help="Override CSV de train: timestamps separados por coma.")
    parser.add_argument("--test-dates", default=None, help="Override CSV de test: timestamps separados por coma.")
    parser.add_argument("--allow-incomplete-split", action="store_true", help="Permite omitir escenas del split sin CSV local.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--tags", default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=None)
    parser.add_argument("--criterion", choices=("gini", "entropy", "log_loss"), default=None)
    parser.add_argument("--splitter", choices=("best", "random"), default=None)
    parser.add_argument("--class-weight", choices=("balanced", "none"), default=None)
    parser.add_argument("--ccp-alpha", type=float, default=None)
    parser.add_argument("--max-features", default=None, help="Número, sqrt, log2 o none.")
    parser.add_argument("--max-leaf-nodes", type=int, default=None)
    parser.add_argument("--min-impurity-decrease", type=float, default=None)
    parser.add_argument("--scene-recall-target", type=float, default=None,
                        help="Recall mínimo exigido por escena con fuego (default 0.9).")
    parser.add_argument("--test-size", type=float, default=None, help="Sólo se usa sin split por fechas.")
    parser.add_argument("--no-stratify", action="store_true")
    parser.add_argument("--no-comet", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).expanduser().resolve().open(encoding="utf-8") as fh:
        value = yaml.safe_load(fh) or {}
    if not isinstance(value, dict):
        raise ValueError(f"La configuración debe ser un mapping YAML: {path}")
    return value


def apply_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    mapping = {
        "features_dir": "features_dir", "feature_sets_config": "feature_sets_config",
        "feature_set": "feature_set", "target": "target", "split_dir": "split_dir",
        "split_id": "split_id", "output_dir": "output_dir", "experiment_name": "experiment_name",
        "tags": "tags", "random_state": "random_state", "max_depth": "max_depth",
        "min_samples_leaf": "min_samples_leaf", "criterion": "criterion", "splitter": "splitter",
        "class_weight": "class_weight", "ccp_alpha": "ccp_alpha", "max_features": "max_features",
        "max_leaf_nodes": "max_leaf_nodes", "min_impurity_decrease": "min_impurity_decrease",
    }
    for key, dest in mapping.items():
        if getattr(args, dest) is None and config.get(key) is not None:
            value = config[key]
            if dest == "tags" and isinstance(value, (list, tuple)):
                value = ",".join(str(item) for item in value)
            setattr(args, dest, value)
    split_cfg = config.get("train_test_split", {})
    if isinstance(split_cfg, dict):
        if args.test_size is None:
            args.test_size = split_cfg.get("test_size", 0.2)
        if not args.no_stratify and split_cfg.get("stratify") is False:
            args.no_stratify = True
    args.features_dir = args.features_dir or str(DEFAULT_FEATURES_DIR)
    args.feature_sets_config = args.feature_sets_config or str(DEFAULT_FEATURE_SETS)
    args.target = args.target or "ref_fire"
    args.output_dir = args.output_dir or str(PROJECT_ROOT / "machine_learning" / "outputs" / "decision_tree")
    args.random_state = 42 if args.random_state is None else args.random_state
    args.max_depth = 6 if args.max_depth is None else args.max_depth
    args.min_samples_leaf = 5 if args.min_samples_leaf is None else args.min_samples_leaf
    args.criterion = args.criterion or "gini"
    args.splitter = args.splitter or "best"
    args.class_weight = args.class_weight or "balanced"
    args.ccp_alpha = 0.0 if args.ccp_alpha is None else args.ccp_alpha
    args.min_impurity_decrease = 0.0 if args.min_impurity_decrease is None else args.min_impurity_decrease
    args.test_size = 0.2 if args.test_size is None else args.test_size
    if args.scene_recall_target is None:
        args.scene_recall_target = float(config.get("scene_recall_target", 0.9))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_dates(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def load_split(args: argparse.Namespace) -> tuple[list[str], list[str], dict[str, Any]]:
    if args.train_dates is not None or args.test_dates is not None:
        train = parse_dates(args.train_dates) or []
        test = parse_dates(args.test_dates) or []
        metadata = {"split_id": "cli_dates", "source": "CLI"}
    else:
        if not args.split_dir:
            raise ValueError("Hace falta --split-dir/--split-id o --train-dates y --test-dates.")
        split_dir = Path(args.split_dir).expanduser().resolve()
        candidates = sorted(split_dir.glob("*_metadata.json"))
        if args.split_id:
            candidates = [p for p in candidates if p.stem.startswith(f"{args.split_id}_")]
        if not candidates:
            raise FileNotFoundError(f"No se encontró metadata para split_id={args.split_id!r} en {split_dir}")
        metadata_path = candidates[0]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        split_id = metadata.get("split_id", metadata_path.stem.replace("_metadata", ""))
        train_path = split_dir / metadata.get("train_file", f"{split_id}_train_dates.csv")
        test_path = split_dir / metadata.get("test_file", f"{split_id}_test_dates.csv")
        train = pd.read_csv(train_path)["date"].astype(str).tolist()
        test = pd.read_csv(test_path)["date"].astype(str).tolist()
        metadata["metadata_file"] = str(metadata_path)
        metadata["train_file_resolved"] = str(train_path)
        metadata["test_file_resolved"] = str(test_path)

    train, test = sorted(set(train)), sorted(set(test))
    overlap = sorted(set(train) & set(test))
    if overlap:
        raise ValueError(f"Train/test comparten timestamps: {overlap}")
    if not train or not test:
        raise ValueError("Train y test deben tener al menos una escena cada uno.")
    return train, test, metadata


def read_scene(path: Path, timestamp: str, target: str) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, engine="pyarrow")
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Formato de features no soportado: {path}")
    required = {"timestamp", target}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: faltan columnas obligatorias {missing}")
    values = frame["timestamp"].dropna().astype(str).unique().tolist()
    if values != [timestamp]:
        raise ValueError(f"{path.name}: timestamp interno inesperado {values}; esperaba {timestamp}")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame[frame[target].isin([0, 1])].copy()
    if frame.empty:
        raise ValueError(f"{path.name}: no quedaron filas con target binario válido")
    return frame


def resolve_scene_path(features_dir: Path, timestamp: str) -> Path | None:
    """Prefer Parquet and fall back to the legacy CSV representation."""
    parquet = features_dir / f"{timestamp}.parquet"
    csv = features_dir / f"{timestamp}.csv"
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    return None


def resolve_features(frame: pd.DataFrame, feature_set: str | None, feature_sets: dict[str, Any], explicit: list[str] | None, target: str) -> list[str]:
    if explicit:
        columns = explicit
    elif not feature_set or feature_set.lower() == "all":
        columns = [c for c in frame.columns if c not in PROTECTED_COLUMNS]
    else:
        columns = []
        for name in feature_set.split("+"):
            name = name.strip()
            if name not in feature_sets or not isinstance(feature_sets[name], list):
                raise ValueError(f"Feature set desconocido: {name}. Disponibles: {sorted(feature_sets)}")
            columns.extend(str(c) for c in feature_sets[name])
    unique = list(dict.fromkeys(columns))
    protected = sorted(set(unique) & PROTECTED_COLUMNS)
    if protected:
        raise ValueError(f"No se permiten como features columnas protegidas: {protected}")
    missing = sorted(set(unique) - set(frame.columns))
    if missing:
        raise ValueError(f"Features no presentes en los CSV: {missing}")
    if target in unique:
        raise ValueError(f"El target {target!r} no puede ser feature")
    if not unique:
        raise ValueError("El feature set no contiene columnas")
    return unique


def build_pipeline(features: list[str], args: argparse.Namespace) -> Pipeline:
    model_kwargs: dict[str, Any] = {
        "criterion": args.criterion,
        "splitter": args.splitter,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "class_weight": None if args.class_weight == "none" else args.class_weight,
        "ccp_alpha": args.ccp_alpha,
        "max_leaf_nodes": args.max_leaf_nodes,
        "min_impurity_decrease": args.min_impurity_decrease,
        "random_state": args.random_state,
    }
    if args.max_features not in (None, "none"):
        model_kwargs["max_features"] = int(args.max_features) if str(args.max_features).isdigit() else args.max_features
    else:
        model_kwargs["max_features"] = None
    preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median", keep_empty_features=True), features)],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", DecisionTreeClassifier(**model_kwargs))])


def metric_values(y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray | None, prefix: str) -> dict[str, float]:
    values = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if probabilities is not None and y_true.nunique() == 2:
        values[f"{prefix}_average_precision"] = float(average_precision_score(y_true, probabilities))
    return values


def confusion_parts(y_true: pd.Series, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tn, fp, fn, tp) forcing both labels so single-class scenes work."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def scene_scores(y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Per-scene metrics where undefined values are NaN instead of a misleading 0.

    A scene without reference fire has no recall to speak of, and a scene where
    the model predicted nothing has no precision.  Collapsing those cases to 0
    would drag every macro average down.
    """
    tn, fp, fn, tp = confusion_parts(y_true, y_pred)
    positives, predicted = tp + fn, tp + fp
    recall = tp / positives if positives else math.nan
    precision = tp / predicted if predicted else math.nan
    if not positives and not predicted:
        f1 = math.nan
    elif math.isnan(recall) or math.isnan(precision) or (recall + precision) == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "rows": int(len(y_true)),
        "positive_rows": positives,
        "predicted_positive_rows": predicted,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "has_fire": bool(positives),
        "alarm_raised": bool(predicted),
        "metrics": {
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "average_precision": float(average_precision_score(y_true, probabilities)) if positives else math.nan,
        },
    }


def scene_aggregates(per_scene: dict[str, dict[str, Any]], recall_target: float) -> dict[str, float]:
    """Scene-level view: every timestamp weighs the same, regardless of pixel count."""
    scenes = list(per_scene.values())
    fire = [s for s in scenes if s["has_fire"]]
    clear = [s for s in scenes if not s["has_fire"]]
    recalls = [s["metrics"]["recall"] for s in fire]
    precisions = [s["metrics"]["precision"] for s in scenes if s["predicted_positive_rows"]]
    false_positives = [s["false_positives"] for s in scenes]

    detected = sum(1 for s in fire if s["alarm_raised"])
    # Stricter: an alarm somewhere in a burning scene is not the same as hitting
    # the burning pixel, so count only scenes with at least one true positive.
    hit = sum(1 for s in fire if s["true_positives"] > 0)
    false_alarm_scenes = sum(1 for s in clear if s["alarm_raised"])
    alarm_recall = detected / len(fire) if fire else math.nan
    alarm_precision = detected / (detected + false_alarm_scenes) if (detected + false_alarm_scenes) else math.nan
    if math.isnan(alarm_recall) or math.isnan(alarm_precision) or (alarm_recall + alarm_precision) == 0:
        alarm_f1 = math.nan
    else:
        alarm_f1 = 2 * alarm_precision * alarm_recall / (alarm_precision + alarm_recall)

    aggregates = {
        "scene_count": float(len(scenes)),
        "scene_count_with_fire": float(len(fire)),
        "scene_count_without_fire": float(len(clear)),
        "scene_macro_recall": float(np.mean(recalls)) if recalls else math.nan,
        "scene_macro_precision": float(np.mean(precisions)) if precisions else math.nan,
        "scene_worst_recall": float(np.min(recalls)) if recalls else math.nan,
        "scene_fraction_meeting_recall_target": (
            float(np.mean([r >= recall_target for r in recalls])) if recalls else math.nan
        ),
        "scene_recall_target": float(recall_target),
        "false_positives_per_scene": float(np.mean(false_positives)) if false_positives else math.nan,
        "false_positives_per_scene_median": float(np.median(false_positives)) if false_positives else math.nan,
        "false_positives_per_scene_max": float(np.max(false_positives)) if false_positives else math.nan,
        "false_positives_per_clear_scene": (
            float(np.mean([s["false_positives"] for s in clear])) if clear else math.nan
        ),
        "scene_alarm_recall": float(alarm_recall),
        "scene_alarm_precision": float(alarm_precision),
        "scene_alarm_f1": float(alarm_f1),
        "scene_alarm_false_alarm_scenes": float(false_alarm_scenes),
        "scene_alarm_missed_scenes": float(len(fire) - detected),
        "scene_hit_recall": float(hit / len(fire)) if fire else math.nan,
        "scene_hit_missed_scenes": float(len(fire) - hit),
    }
    return aggregates


def finite_only(values: dict[str, float]) -> dict[str, float]:
    """Comet rejects NaN, so undefined metrics are dropped instead of logged as 0."""
    return {k: v for k, v in values.items() if not (isinstance(v, float) and not math.isfinite(v))}


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def strip_nan(value: Any) -> Any:
    """NaN is not valid JSON, so undefined metrics are persisted as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: strip_nan(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strip_nan(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(strip_nan(value), indent=2, sort_keys=True, default=str)
    path.write_text(payload + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.experiment_config)
    apply_config(args, config)
    if args.no_comet:
        os.environ["COMET_ENABLED"] = "false"
    if args.tags:
        os.environ["COMET_TAGS"] = args.tags if isinstance(args.tags, str) else ",".join(args.tags)

    features_dir = Path(args.features_dir).expanduser().resolve()
    feature_sets = load_yaml(args.feature_sets_config)
    train_dates, test_dates, split_metadata = load_split(args)
    all_dates = train_dates + test_dates
    loaded: dict[str, pd.DataFrame] = {}
    manifest: list[dict[str, Any]] = []
    missing_scenes: list[str] = []
    for timestamp in all_dates:
        path = resolve_scene_path(features_dir, timestamp)
        if path is None:
            missing_scenes.append(timestamp)
            continue
        frame = read_scene(path, timestamp, args.target)
        loaded[timestamp] = frame
        manifest.append({
            "timestamp": timestamp,
            "split": "train" if timestamp in train_dates else "test",
            "file": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
            "format": path.suffix.lstrip("."),
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
            "positive_rows": int(frame[args.target].sum()),
        })
    if missing_scenes and not args.allow_incomplete_split:
        raise FileNotFoundError(
            f"Faltan {len(missing_scenes)} CSVs del split. Primeros: {missing_scenes[:10]}. "
            "Descárgalos/genera features o usa --allow-incomplete-split explícitamente."
        )
    train_dates = [d for d in train_dates if d in loaded]
    test_dates = [d for d in test_dates if d in loaded]
    if not train_dates or not test_dates:
        raise ValueError("Después de aplicar cobertura local, train o test quedó sin escenas.")

    reference = loaded[train_dates[0]]
    features = resolve_features(reference, args.feature_set, feature_sets, args.feature_columns, args.target)
    train = pd.concat([loaded[d] for d in train_dates], ignore_index=True)
    test = pd.concat([loaded[d] for d in test_dates], ignore_index=True)
    features = [c for c in features if train[c].notna().any()]
    if not features:
        raise ValueError("Todas las features seleccionadas están completamente vacías en train.")
    dropped_all_missing = sorted(set(resolve_features(reference, args.feature_set, feature_sets, args.feature_columns, args.target)) - set(features))
    X_train, y_train = train[features].apply(pd.to_numeric, errors="coerce"), train[args.target].astype(int)
    X_test, y_test = test[features].apply(pd.to_numeric, errors="coerce"), test[args.target].astype(int)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(f"Se necesitan ambas clases en train y test: train={sorted(y_train.unique())}, test={sorted(y_test.unique())}")

    manifest_payload = {"features_dir": str(features_dir), "entries": manifest, "missing_scenes": missing_scenes}
    manifest_hash = hashlib.sha256(json.dumps(manifest_payload, sort_keys=True).encode()).hexdigest()
    run_spec = {
        "feature_set": args.feature_set or "explicit",
        "feature_columns": features,
        "target": args.target,
        "manifest_hash": manifest_hash,
        "hyperparameters": {
            "random_state": args.random_state,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "criterion": args.criterion,
            "splitter": args.splitter,
            "class_weight": args.class_weight,
            "ccp_alpha": args.ccp_alpha,
            "max_features": args.max_features,
            "max_leaf_nodes": args.max_leaf_nodes,
            "min_impurity_decrease": args.min_impurity_decrease,
        },
    }
    spec_hash = hashlib.sha256(json.dumps(run_spec, sort_keys=True, default=str).encode()).hexdigest()[:10]
    label = args.experiment_name or args.feature_set or "explicit"
    label = "".join(char if char.isalnum() or char in "-_" else "-" for char in label).strip("-")
    run_id = f"dt_{label}_{split_metadata.get('split_id', 'split')}_{spec_hash}"
    output_dir = Path(args.output_dir).expanduser().resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(features, args)
    tracker = create_tracker(experiment_name=args.experiment_name or run_id)
    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_probability = pipeline.predict_proba(X_test)[:, 1]
        metrics = metric_values(y_test, y_pred, y_probability, "test")
        metrics.update(metric_values(y_train, pipeline.predict(X_train), None, "train"))
        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

        per_scene: dict[str, dict[str, Any]] = {}
        for timestamp, group in test.groupby("timestamp", sort=True):
            scene_X = group[features].apply(pd.to_numeric, errors="coerce")
            scene_y = group[args.target].astype(int)
            scene_pred = pipeline.predict(scene_X)
            scene_prob = pipeline.predict_proba(scene_X)[:, 1]
            per_scene[timestamp] = scene_scores(scene_y, scene_pred, scene_prob)

        aggregates = scene_aggregates(per_scene, args.scene_recall_target)
        metrics.update(aggregates)
        for timestamp, scene in per_scene.items():
            for name, value in scene["metrics"].items():
                metrics[f"scene_{timestamp}_{name}"] = value
            metrics[f"scene_{timestamp}_false_positives"] = float(scene["false_positives"])

        model = pipeline.named_steps["model"]
        importances = {
            feature: float(value)
            for feature, value in sorted(zip(features, model.feature_importances_), key=lambda item: -item[1])
        }
        resolved = {
            "model": "DecisionTreeClassifier",
            "feature_set": args.feature_set or "explicit",
            "feature_columns": features,
            "dropped_all_missing_from_train": dropped_all_missing,
            "target": args.target,
            "train_dates": train_dates,
            "test_dates": test_dates,
            "missing_scenes": missing_scenes,
            "split": split_metadata,
            "dataset": {
                "features_dir": str(features_dir),
                "manifest_sha256": manifest_hash,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_positive_rows": int(y_train.sum()),
                "test_positive_rows": int(y_test.sum()),
            },
            "hyperparameters": {
                "random_state": args.random_state,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "criterion": args.criterion,
                "splitter": args.splitter,
                "class_weight": args.class_weight,
                "ccp_alpha": args.ccp_alpha,
                "max_features": args.max_features,
                "max_leaf_nodes": args.max_leaf_nodes,
                "min_impurity_decrease": args.min_impurity_decrease,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "git_revision": git_revision(),
            },
        }
        write_json(output_dir / "config_resolved.json", resolved)
        write_json(output_dir / "dataset_manifest.json", manifest_payload)
        write_json(output_dir / "metrics.json", {
            "global": metrics,
            "confusion_matrix": matrix,
            "scene_level": aggregates,
            "per_scene": per_scene,
        })
        write_json(output_dir / "feature_importances.json", importances)
        joblib.dump(pipeline, output_dir / "model.joblib")

        tracker.log_parameters({
            "model": "DecisionTreeClassifier",
            "feature_set": args.feature_set or "explicit",
            "feature_columns": ",".join(features),
            "dropped_all_missing_features": ",".join(dropped_all_missing) or "none",
            "target": args.target,
            "split_id": split_metadata.get("split_id", "cli_dates"),
            "dataset_features_dir": str(features_dir),
            "dataset_manifest_sha256": manifest_hash,
            "dataset_manifest_file": str(output_dir / "dataset_manifest.json"),
            "train_dates": ",".join(train_dates),
            "test_dates": ",".join(test_dates),
            "missing_scenes": ",".join(missing_scenes) or "none",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_positive_rows": int(y_train.sum()),
            "test_positive_rows": int(y_test.sum()),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "random_state": int(args.random_state),
            "max_depth": args.max_depth,
            "min_samples_leaf": int(args.min_samples_leaf),
            "criterion": args.criterion,
            "splitter": args.splitter,
            "class_weight": args.class_weight,
            "ccp_alpha": float(args.ccp_alpha),
            "scene_recall_target": float(args.scene_recall_target),
            "test_scenes_with_fire": int(aggregates["scene_count_with_fire"]),
            "test_scenes_without_fire": int(aggregates["scene_count_without_fire"]),
            "git_revision": resolved["environment"]["git_revision"],
        })
        tracker.log_metrics(finite_only(metrics), step=1)
        tracker.log_confusion_matrix(y_test.to_numpy(), y_pred, labels=[0, 1])
        for asset in ("config_resolved.json", "dataset_manifest.json", "metrics.json", "feature_importances.json", "model.joblib"):
            tracker.log_asset(output_dir / asset, file_name=f"{run_id}/{asset}")
    finally:
        tracker.end()

    print(f"Run: {run_id}")
    print(f"Output: {output_dir}")
    print(f"Features ({len(features)}): {features}")
    print(f"Train scenes/rows: {len(train_dates)}/{len(train)} | positives={int(y_train.sum())}")
    print(f"Test scenes/rows: {len(test_dates)}/{len(test)} | positives={int(y_test.sum())}")

    def fmt(value: float) -> str:
        return "  --  " if isinstance(value, float) and not math.isfinite(value) else f"{value:6.4f}"

    print("\nPor escena (test):")
    print(f"  {'timestamp':<16}{'pos':>5}{'TP':>5}{'FN':>5}{'FP':>6}  {'recall':>7} {'precision':>9}  alarma")
    for timestamp, scene in per_scene.items():
        alarm = "sí" if scene["alarm_raised"] else "no"
        print(
            f"  {timestamp:<16}{scene['positive_rows']:>5}{scene['true_positives']:>5}"
            f"{scene['false_negatives']:>5}{scene['false_positives']:>6}  "
            f"{fmt(scene['metrics']['recall']):>7} {fmt(scene['metrics']['precision']):>9}  {alarm}"
        )

    print("\nNivel escena (cada timestamp pesa igual):")
    print(f"  escenas con fuego / sin fuego : {int(aggregates['scene_count_with_fire'])} / {int(aggregates['scene_count_without_fire'])}")
    print(f"  recall macro (escenas c/fuego): {fmt(aggregates['scene_macro_recall'])}")
    print(f"  peor recall de una escena     : {fmt(aggregates['scene_worst_recall'])}")
    print(f"  escenas que cumplen recall>={aggregates['scene_recall_target']:.2f}: {fmt(aggregates['scene_fraction_meeting_recall_target'])}")
    print(f"  precision macro               : {fmt(aggregates['scene_macro_precision'])}")
    print(f"  FP por escena (media/mediana/max): {aggregates['false_positives_per_scene']:.2f} / "
          f"{aggregates['false_positives_per_scene_median']:.2f} / {int(aggregates['false_positives_per_scene_max'])}")
    print(f"  FP por escena sin fuego       : {fmt(aggregates['false_positives_per_clear_scene'])}")
    print(f"  alarma por escena r/p/f1      : {fmt(aggregates['scene_alarm_recall'])} / "
          f"{fmt(aggregates['scene_alarm_precision'])} / {fmt(aggregates['scene_alarm_f1'])}")
    print(f"  escenas con acierto real (TP>0): {fmt(aggregates['scene_hit_recall'])}")
    print(f"  escenas con fuego sin acierto : {int(aggregates['scene_hit_missed_scenes'])}")
    print(f"  escenas sin fuego con alarma  : {int(aggregates['scene_alarm_false_alarm_scenes'])}")

    print("\nAgregado por pixel (micro, todas las escenas juntas):")
    for name in ("test_precision", "test_recall", "test_f1", "test_average_precision"):
        if name in metrics:
            print(f"  {name}: {metrics[name]:.4f}")
    print(f"  confusion_matrix [[TN,FP],[FN,TP]]: {matrix}")


if __name__ == "__main__":
    main()
