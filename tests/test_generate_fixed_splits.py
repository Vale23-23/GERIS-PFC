from pathlib import Path

import numpy as np

from machine_learning.experiments.generate_fixed_splits import (
    generate_scene_split,
    resample_b02_to_grid,
    write_split_bundle,
)


def test_generate_scene_split_uses_date_level_fixed_split(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "ABI-L1b-Rad-B07").mkdir(parents=True)
    (dataset_root / "ABI-L2-FDCF-Mask").mkdir(parents=True)

    dates = ["20251115_1500", "20251115_1530", "20251115_1600", "20251115_1630"]
    for date in dates:
        np.save(dataset_root / "ABI-L1b-Rad-B07" / f"{date}.npy", np.ones((4, 4), dtype=np.float32))

    fire_mask_values = [
        np.array([[10, 100], [100, 100]], dtype=np.int16),
        np.array([[100, 100], [100, 100]], dtype=np.int16),
        np.array([[10, 10], [100, 100]], dtype=np.int16),
        np.array([[100, 100], [100, 100]], dtype=np.int16),
    ]

    for date, values in zip(dates, fire_mask_values):
        np.save(dataset_root / "ABI-L2-FDCF-Mask" / f"{date}.npy", values)

    train_dates, test_dates = generate_scene_split(dataset_root, test_size=0.5, random_state=42)

    assert set(train_dates).union(test_dates) == set(dates)
    assert set(train_dates).isdisjoint(test_dates)
    assert len(train_dates) + len(test_dates) == len(dates)


def test_generate_scene_split_can_limit_number_of_scenes(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "ABI-L1b-Rad-B07").mkdir(parents=True)
    (dataset_root / "ABI-L2-FDCF-Mask").mkdir(parents=True)

    dates = [f"20251115_{hour:04d}" for hour in range(1500, 1500 + 8 * 30, 30)]
    for date in dates:
        np.save(dataset_root / "ABI-L1b-Rad-B07" / f"{date}.npy", np.ones((4, 4), dtype=np.float32))

    for date in dates:
        values = np.full((4, 4), 100, dtype=np.int16)
        if date.endswith("1500"):
            values[0, 0] = 10
        np.save(dataset_root / "ABI-L2-FDCF-Mask" / f"{date}.npy", values)

    train_dates, test_dates = generate_scene_split(
        dataset_root,
        test_size=0.5,
        random_state=42,
        max_scenes=4,
    )

    assert len(train_dates) + len(test_dates) == 4
    assert set(train_dates).union(test_dates) <= set(dates)


def test_generated_split_bundle_has_explicit_identifier(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "ABI-L1b-Rad-B07").mkdir(parents=True)
    (dataset_root / "ABI-L2-FDCF-Mask").mkdir(parents=True)

    dates = ["20251115_1500", "20251115_1530", "20251115_1600", "20251115_1630"]
    for date in dates:
        np.save(dataset_root / "ABI-L1b-Rad-B07" / f"{date}.npy", np.ones((4, 4), dtype=np.float32))

    for date in dates:
        values = np.full((4, 4), 100, dtype=np.int16)
        if date == "20251115_1500":
            values[0, 0] = 10
        np.save(dataset_root / "ABI-L2-FDCF-Mask" / f"{date}.npy", values)

    output_dir = tmp_path / "splits"
    train_dates, test_dates = generate_scene_split(dataset_root, test_size=0.5, random_state=42)
    split_id = "scene_split_test_run"
    train_path, test_path, metadata_path = write_split_bundle(
        output_dir,
        train_dates,
        test_dates,
        split_id=split_id,
        dataset_root=dataset_root,
        test_size=0.5,
        random_state=42,
    )

    assert split_id in train_path.name
    assert split_id in test_path.name
    assert metadata_path.exists()
    assert metadata_path.name.startswith(f"{split_id}_")


def test_resample_b02_to_grid_reduces_thermal_resolution():
    rad = np.arange(16, dtype=np.float32).reshape(4, 4)

    result = resample_b02_to_grid(rad, target_shape=(2, 2), max_pixel_slack=4)

    expected = np.array([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_generate_scene_split_keeps_fire_in_both_sets_when_possible(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "ABI-L1b-Rad-B07").mkdir(parents=True)
    (dataset_root / "ABI-L2-FDCF-Mask").mkdir(parents=True)

    dates = [f"20251115_{hour:04d}" for hour in range(1500, 1500 + 8 * 30, 30)]
    for date in dates:
        np.save(dataset_root / "ABI-L1b-Rad-B07" / f"{date}.npy", np.ones((4, 4), dtype=np.float32))

    for idx, date in enumerate(dates):
        values = np.full((4, 4), 100, dtype=np.int16)
        if idx < 4:
            values[0, 0] = 10
        np.save(dataset_root / "ABI-L2-FDCF-Mask" / f"{date}.npy", values)

    train_dates, test_dates = generate_scene_split(
        dataset_root,
        test_size=0.5,
        random_state=42,
        min_positive_scenes=1,
    )

    train_fire = [date for date in train_dates if date in dates[:4]]
    test_fire = [date for date in test_dates if date in dates[:4]]
    train_no_fire = [date for date in train_dates if date in dates[4:]]
    test_no_fire = [date for date in test_dates if date in dates[4:]]

    assert train_fire and test_fire
    assert train_no_fire and test_no_fire
