import argparse
import importlib.util
from pathlib import Path

import numpy as np

from fdca import run_audit

PLOT_SCRIPT = Path(__file__).resolve().parents[1] / "implementacion" / "plot_audit_pixels.py"
spec = importlib.util.spec_from_file_location("plot_audit_pixels", PLOT_SCRIPT)
plot_audit_pixels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_audit_pixels)


def test_ensure_selected_timestamps_downloads_each_scene(monkeypatch, tmp_path):
    calls = []

    def fake_ensure_timestamp_data(timestamp, region, dataset_root, download, repo_id=None):
        calls.append((timestamp, region, str(dataset_root), download, repo_id))

    monkeypatch.setattr(run_audit, "ensure_timestamp_data", fake_ensure_timestamp_data)

    args = argparse.Namespace(
        region="uruguay",
        dataset_root=str(tmp_path),
        download=True,
    )

    run_audit.ensure_selected_timestamps(args, ["20251115_1500", "20251116_1200"])

    assert calls == [
        ("20251115_1500", "uruguay", str(tmp_path), True, run_audit.DEFAULT_HF_REPO_ID),
        ("20251116_1200", "uruguay", str(tmp_path), True, run_audit.DEFAULT_HF_REPO_ID),
    ]


def test_pick_timestamps_uses_remote_reference_listing(monkeypatch, tmp_path):
    remote_timestamps = [
        "20251115_1500",
        "20251116_1200",
        "20251117_1200",
        "20251118_1200",
    ]

    monkeypatch.setattr(
        run_audit,
        "list_remote_reference_timestamps",
        lambda region, repo_id: remote_timestamps,
    )

    args = argparse.Namespace(
        timestamps=None,
        n=2,
        seed=42,
        only_with_fire=False,
        region="uruguay",
        dataset_root=str(tmp_path),
        download=True,
    )

    selected = run_audit.pick_timestamps(args)

    assert len(selected) == 2
    assert set(selected).issubset(set(remote_timestamps))


def test_scene_metrics_ignores_pixels_out_of_region():
    reference = np.array([[0, 10], [10, 0]], dtype=np.int32)
    prediction = np.array([[0, 0], [10, 0]], dtype=np.int32)
    candidate_mask = np.array([[False, False], [False, False]], dtype=bool)
    region_mask = np.array([[True, False], [True, True]], dtype=bool)

    metrics = run_audit.scene_metrics(
        reference=reference,
        candidate_mask=candidate_mask,
        fire_mask_p2=prediction,
        region_mask=region_mask,
    )

    assert metrics["part2_binary"]["tp"] == 1
    assert metrics["part2_binary"]["fn"] == 0
    assert metrics["part2_binary"]["fp"] == 0
    assert metrics["part2_binary"]["support_ref"] == 1


def test_scene_metrics_ignores_low_probability_code_15():
    reference = np.array([[15, 10], [0, 35]], dtype=np.int32)
    prediction = np.array([[15, 10], [0, 10]], dtype=np.int32)

    metrics = run_audit.scene_metrics(
        reference=reference,
        candidate_mask=np.zeros_like(reference, dtype=bool),
        fire_mask_p2=prediction,
        region_mask=np.ones_like(reference, dtype=bool),
    )

    assert metrics["part2_binary"]["tp"] == 3
    assert metrics["part2_binary_no_low_prob"]["tp"] == 1
    assert metrics["part2_binary_no_low_prob"]["fn"] == 0
    assert metrics["part2_binary_no_low_prob"]["fp"] == 0
    assert metrics["part2_binary_no_low_prob"]["support_ref"] == 1
    assert "15" not in metrics["by_base_code_no_low_prob"]


def test_plot_data_ignores_pixels_out_of_roi():
    rows = [
        {"lat": "-34.0", "lon": "-56.0", "verdict": "FN", "eco_code": "150", "in_roi": "1"},
        {"lat": "-35.0", "lon": "-58.0", "verdict": "FN", "eco_code": "150", "in_roi": "0"},
        {"lat": "-34.1", "lon": "-56.1", "verdict": "FP", "eco_code": "0", "in_roi": "0"},
        {"lat": "-33.9", "lon": "-55.9", "verdict": "FP", "eco_code": "0", "in_roi": "1"},
    ]

    eco_points, fn_points, fp_points = plot_audit_pixels.build_plot_data(rows)

    assert len(eco_points) == 1
    assert len(fn_points) == 1
    assert len(fp_points) == 1
    assert fn_points[0] == (-34.0, -56.0)
    assert fp_points[0] == (-33.9, -55.9)
