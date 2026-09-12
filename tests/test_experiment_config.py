import argparse
import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from machine_learning.experiments.decision_tree_real_data import apply_experiment_config

SCRIPT_PATH = PROJECT_ROOT / "machine_learning" / "experiments" / "decision_tree_v1"
loader = SourceFileLoader("decision_tree_v1_module", str(SCRIPT_PATH))
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)
apply_scene_experiment_config = module.apply_experiment_config


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


def test_scene_experiment_config_is_applied_from_yaml():
    args = argparse.Namespace(
        dataset_root="/tmp/data",
        dates=None,
        split_dir=None,
        split_id=None,
        experiment_name=None,
        tags="goes,b02-b07-b14,tree",
        max_samples_per_class=2000,
        test_size=0.2,
        stratify=True,
        random_state=42,
        max_depth=5,
        min_samples_leaf=20,
        config=None,
        no_comet=False,
    )

    experiment_config = {
        "model": "DecisionTreeClassifier",
        "random_state": 7,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "train_test_split": {
            "test_size": 0.25,
            "stratify": False,
        },
        "experiment_name": "dt_baseline_v1",
        "tags": ["baseline", "fixed_split", "yaml_config"],
    }

    apply_scene_experiment_config(args, experiment_config)

    assert args.random_state == 7
    assert args.max_depth == 6
    assert args.min_samples_leaf == 5
    assert args.test_size == 0.25
    assert args.stratify is False
    assert args.experiment_name == "dt_baseline_v1"
    assert args.tags == "baseline,fixed_split,yaml_config"
