import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.tracking import create_tracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a decision tree on a CSV dataset with fixed splits and feature-set definitions.")
    parser.add_argument("--csv", required=True, help="Path to the CSV dataset to use.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "machine_learning" / "configs" / "feature_sets.yaml"), help="YAML file with named feature sets.")
    parser.add_argument("--experiment-config", default=None, help="Optional YAML experiment file with model hyperparameters and feature-set names.")
    parser.add_argument("--feature-set", default="all", help="Named feature set from the YAML, or 'all' to use every non-target column.")
    parser.add_argument("--feature-columns", nargs="*", default=None, help="Optional explicit list of columns to use as features.")
    parser.add_argument("--train-split", default=None, help="CSV containing a single 'row_id' column for training rows.")
    parser.add_argument("--test-split", default=None, help="CSV containing a single 'row_id' column for test rows.")
    parser.add_argument("--experiment-name", default=os.getenv("COMET_EXPERIMENT_NAME"), help="Comet experiment name.")
    parser.add_argument("--tags", default=os.getenv("COMET_TAGS", "real-data,decision-tree"), help="Comma-separated Comet tags.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction of the dataset to reserve for testing when no split files are provided.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum depth of the decision tree.")
    parser.add_argument("--min-samples-leaf", type=int, default=5, help="Minimum samples per leaf.")
    parser.add_argument("--drop-columns", nargs="*", default=[], help="Optional columns to drop before modeling.")
    parser.add_argument("--no-comet", action="store_true", help="Disable Comet logging for local-only runs.")
    return parser.parse_args()


def load_yaml_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if isinstance(data, dict):
        return data
    raise ValueError(f"YAML config at '{config_path}' must contain a dictionary at the root.")


def load_feature_sets(path: str | None) -> dict:
    return load_yaml_config(path)


def resolve_feature_columns(df: pd.DataFrame, target: str, feature_set_name: str, feature_sets: dict, explicit_columns: list[str] | None) -> list[str]:
    if explicit_columns:
        available = [col for col in explicit_columns if col in df.columns]
        missing = [col for col in explicit_columns if col not in df.columns]
        if missing:
            raise ValueError(f"The following feature columns were not found: {missing}")
        return available

    if feature_set_name and feature_set_name.lower() != "all":
        names = [part.strip() for part in feature_set_name.split("+") if part.strip()]
        if names:
            chosen_columns: list[str] = []
            for name in names:
                if name not in feature_sets:
                    raise ValueError(f"Unknown feature set '{name}'. Available sets: {sorted(feature_sets)}")
                chosen_columns.extend(feature_sets[name])
            unique = []
            for col in chosen_columns:
                if col not in unique:
                    unique.append(col)
            missing = [col for col in unique if col not in df.columns]
            if missing:
                raise ValueError(f"Columns referenced by feature set are missing from the CSV: {missing}")
            return unique

    return [col for col in df.columns if col != target]


def load_split_ids(path: str | None) -> list[int] | None:
    if not path:
        return None
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Split file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "row_id" not in df.columns:
        raise ValueError(f"Split file '{csv_path}' must contain a 'row_id' column.")
    return df["row_id"].astype(int).tolist()


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str], max_depth: int, min_samples_leaf: int, random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    model = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    args = parse_args()
    experiment_config = load_yaml_config(args.experiment_config)
    if experiment_config:
        for key in [
            "feature_set",
            "target",
            "test_size",
            "random_state",
            "max_depth",
            "min_samples_leaf",
            "tags",
            "experiment_name",
        ]:
            if key in experiment_config and experiment_config[key] is not None:
                if key == "feature_set" and args.feature_set != "all":
                    continue
                setattr(args, key, experiment_config[key])

    if isinstance(args.tags, list):
        args.tags = ",".join(str(tag) for tag in args.tags)

    if args.no_comet:
        os.environ["COMET_ENABLED"] = "false"

    dataset_path = Path(args.csv).expanduser().resolve()
    df = pd.read_csv(dataset_path)

    drop_candidates = [col for col in args.drop_columns if col in df.columns]
    working = df.drop(columns=drop_candidates, errors="ignore")

    if args.target not in working.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV.")

    feature_sets = load_feature_sets(args.config)
    feature_columns = resolve_feature_columns(working, args.target, args.feature_set, feature_sets, args.feature_columns)
    X = working[feature_columns].copy()
    y = working[args.target].copy()

    if X.empty:
        raise ValueError("No feature columns remain after resolving the feature set.")
    if y.nunique() < 2:
        raise ValueError("Target column must have at least two classes for classification.")

    train_ids = load_split_ids(args.train_split)
    test_ids = load_split_ids(args.test_split)

    if train_ids is not None and test_ids is not None:
        if len(set(train_ids).intersection(test_ids)) > 0:
            raise ValueError("Train and test split IDs overlap. They must be disjoint.")
        X_train = X.iloc[train_ids].copy()
        y_train = y.iloc[train_ids].copy()
        X_test = X.iloc[test_ids].copy()
        y_test = y.iloc[test_ids].copy()
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    pipeline = build_pipeline(numeric_cols, categorical_cols, args.max_depth, args.min_samples_leaf, args.random_state)

    experiment_name = args.experiment_name or os.getenv("COMET_EXPERIMENT_NAME") or (
        f"dt_{args.feature_set}_{dataset_path.stem}_{args.max_depth}d"
    )
    os.environ["COMET_EXPERIMENT_NAME"] = experiment_name
    if args.tags:
        os.environ["COMET_TAGS"] = args.tags

    tracker = create_tracker(experiment_name=experiment_name)
    tracker.log_parameters(
        {
            "model": "DecisionTreeClassifier",
            "dataset_path": str(dataset_path),
            "feature_set": args.feature_set,
            "feature_columns": ",".join(feature_columns),
            "target_column": args.target,
            "train_ids_source": str(args.train_split) if args.train_split else "random_split",
            "test_ids_source": str(args.test_split) if args.test_split else "random_split",
            "dataset_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "random_state": int(args.random_state),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "test_size": float(args.test_size),
            "numeric_columns": ",".join(numeric_cols) if numeric_cols else "none",
            "categorical_columns": ",".join(categorical_cols) if categorical_cols else "none",
            "target_classes": ",".join(str(value) for value in sorted(y.unique().tolist())),
        }
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    try:
        cm = confusion_matrix(y_test, y_pred)
        if hasattr(tracker, "experiment"):
            tracker.experiment.log_confusion_matrix(cm, labels=sorted(y.unique().tolist()))
    except Exception:
        pass

    tracker.log_metrics(metrics, step=1)
    tracker.end()

    print("Training complete.")
    print(f"Dataset: {dataset_path}")
    print(f"Feature set: {args.feature_set}")
    print(f"Rows used: train={len(X_train)} test={len(X_test)}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
