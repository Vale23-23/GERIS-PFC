"""Tabla comparativa de corridas (metrics.json) de un directorio de outputs.

Uso: python machine_learning/experiments/compare_runs.py [--output-dir DIR] [--sort METRICA]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLUMNS = {
    "test_average_precision": "AP",
    "test_recall": "recall",
    "test_precision": "prec",
    "test_f1": "f1",
    "train_f1": "train_f1",
    "scene_macro_recall": "sc_recall",
    "scene_worst_recall": "sc_worst",
    "scene_fraction_meeting_recall_target": "sc_meet",
    "scene_macro_precision": "sc_prec",
    "false_positives_per_scene": "FP/scene",
    "scene_alarm_false_alarm_scenes": "FA_scenes",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "machine_learning" / "outputs" / "random_forest"))
    parser.add_argument("--sort", default="test_average_precision", choices=list(COLUMNS))
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(Path(args.output_dir).expanduser().glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        values = {**payload.get("global", {}), **payload.get("scene_level", {})}
        rows.append({"run": metrics_path.parent.name, **{short: values.get(key) for key, short in COLUMNS.items()}})
    if not rows:
        raise SystemExit(f"No hay metrics.json en {args.output_dir}")

    table = pd.DataFrame(rows).set_index("run")
    table = table.sort_values(COLUMNS[args.sort], ascending=False, na_position="last").round(3)
    pd.set_option("display.width", 250, "display.max_columns", None, "display.max_colwidth", 80)
    print(table.to_string())


if __name__ == "__main__":
    main()