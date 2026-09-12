import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.experiments.decision_tree_real_data import apply_experiment_config


def test_apply_experiment_config_reads_nested_train_test_split():
    args = argparse.Namespace(
        feature_set="all",
        target="target",
        test_size=0.25,
        stratify=True,
        random_state=42,
        max_depth=5,
        min_samples_leaf=5,
        tags="real-data,decision-tree",
        experiment_name=None,
        feature_columns=None,
        train_split=None,
        test_split=None,
    )

    experiment_config = {
        "train_test_split": {
            "test_size": 0.3,
            "stratify": False,
        }
    }

    apply_experiment_config(args, experiment_config)

    assert args.test_size == 0.3
    assert args.stratify is False
