"""Optional experiment tracking for the machine-learning workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Protocol

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


class Tracker(Protocol):
    """Small interface used by training code without importing Comet."""

    def log_parameters(self, parameters: Mapping[str, object]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None: ...
    def end(self) -> None: ...


@dataclass
class NullTracker:
    """No-op tracker used when Comet is disabled."""

    def log_parameters(self, parameters: Mapping[str, object]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        return None

    def end(self) -> None:
        return None


class CometTracker:
    """Adapter around a Comet experiment."""

    def __init__(self, experiment: object) -> None:
        self.experiment = experiment

    def log_parameters(self, parameters: Mapping[str, object]) -> None:
        self.experiment.log_parameters(dict(parameters))

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        kwargs = {"step": step} if step is not None else {}
        self.experiment.log_metrics(dict(metrics), **kwargs)

    def end(self) -> None:
        self.experiment.end()


def create_tracker(
    enabled: Optional[bool] = None,
    experiment_name: Optional[str] = None,
) -> Tracker:
    """Create a Comet tracker when explicitly enabled, otherwise a no-op tracker."""
    if enabled is None:
        enabled = os.getenv("COMET_ENABLED", "false").lower() in {"1", "true", "yes", "on"}

    if not enabled:
        return NullTracker()

    if not os.getenv("COMET_API_KEY"):
        raise RuntimeError("COMET_ENABLED is true but COMET_API_KEY is not set")

    try:
        import comet_ml
    except ImportError as exc:
        raise RuntimeError(
            "Comet tracking is enabled but comet_ml is not installed. "
            "Install requirements.txt or machine_learning/requirements-comet.txt."
        ) from exc

    kwargs = {
        "project_name": os.getenv("COMET_PROJECT_NAME", "geris-pfc-ml"),
    }
    workspace = os.getenv("COMET_WORKSPACE")
    if workspace:
        kwargs["workspace"] = workspace

    name = experiment_name or os.getenv("COMET_EXPERIMENT_NAME")

    tags = os.getenv("COMET_TAGS", "")
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    experiment = comet_ml.start(**kwargs)
    if name:
        experiment.set_name(name)
    if tag_list:
        experiment.add_tags(tag_list)
    return CometTracker(experiment)
