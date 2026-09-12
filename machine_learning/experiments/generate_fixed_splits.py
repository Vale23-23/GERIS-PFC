import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BAND_FOLDERS = {
    "B02": "ABI-L1b-Rad-B02",
    "B07": "ABI-L1b-Rad-B07",
    "B14": "ABI-L1b-Rad-B14",
}
FIRE_MASK_FOLDER = "ABI-L2-FDCF-Mask"
FIRE_CODES = {10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed train/test split by GOES scene date for the pixel-wise Decision Tree workflow."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Folder containing the B02/B07/B14 band folders and the FDCF fire-mask folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="machine_learning/splits",
        help="Directory where train/test scene lists will be written.",
    )
    parser.add_argument("--split-id", default=None, help="Optional explicit split identifier. If omitted, one is generated automatically.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction assigned to the test set.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--dates", nargs="*", default=None, help="Optional explicit date list (YYYYMMDD_HHMM).")
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional cap on how many scenes to include before splitting. Useful for quick experiments.",
    )
    parser.add_argument(
        "--min-positive-scenes",
        type=int,
        default=1,
        help="Minimum number of fire-positive scenes kept in each split side when both classes are available.",
    )
    parser.add_argument(
        "--min-negative-scenes",
        type=int,
        default=1,
        help="Minimum number of non-fire scenes kept in each split side when both classes are available.",
    )
    return parser.parse_args()


def discover_dates(dataset_root: Path) -> list[str]:
    reference_folder = dataset_root / BAND_FOLDERS["B07"]
    if not reference_folder.exists():
        raise FileNotFoundError(f"Reference folder not found: {reference_folder}")
    dates = sorted(p.stem for p in reference_folder.glob("*.npy"))
    if not dates:
        raise FileNotFoundError(f"No .npy files found in {reference_folder}")
    return dates


def resample_b02_to_grid(rad: np.ndarray, target_shape: tuple[int, int], max_pixel_slack: int = 4) -> np.ndarray:
    """Reduce B02 to the 2 km thermal grid using area-block averaging."""
    target_h, target_w = target_shape
    height, width = rad.shape

    factor_h = round(height / target_h)
    factor_w = round(width / target_w)
    if factor_h != factor_w or factor_h < 1:
        raise ValueError(
            f"B02 shape {rad.shape} no es compatible con la grilla {target_shape} "
            f"(factor_h={factor_h}, factor_w={factor_w})"
        )
    factor = factor_h
    expected_h, expected_w = target_h * factor, target_w * factor

    pad_h = expected_h - height
    pad_w = expected_w - width
    if pad_h < 0 or pad_w < 0 or max(pad_h, pad_w) > max_pixel_slack:
        raise ValueError(
            f"B02 shape {rad.shape} difiere demasiado de lo esperado "
            f"({expected_h}, {expected_w}) para factor={factor}."
        )
    if pad_h or pad_w:
        rad = np.pad(rad, ((0, pad_h), (0, pad_w)), constant_values=np.nan)

    valid_rad = np.where(rad >= 0, rad, np.nan)
    blocks = valid_rad.reshape(target_h, factor, target_w, factor)

    valid_count = np.sum(np.isfinite(blocks), axis=(1, 3))
    valid_sum = np.nansum(blocks, axis=(1, 3))
    result = np.full((target_h, target_w), np.nan, dtype=np.float64)
    np.divide(valid_sum, valid_count, out=result, where=valid_count > 0)
    return result.astype(np.float32)


def load_scene(dataset_root: Path, date_str: str) -> np.ndarray:
    fire_codes = np.load(dataset_root / FIRE_MASK_FOLDER / f"{date_str}.npy")
    return fire_codes.astype(np.int16)


def codes_to_binary_label(fire_codes: np.ndarray) -> np.ndarray:
    return np.isin(fire_codes, list(FIRE_CODES)).astype(int)


def scene_has_fire(dataset_root: Path, date_str: str) -> bool:
    fire_codes = load_scene(dataset_root, date_str)
    valid_mask = np.isfinite(fire_codes)
    if not np.any(valid_mask):
        return False
    y = codes_to_binary_label(fire_codes[valid_mask])
    return bool(np.any(y == 1))


def split_scene_class(
    class_dates: list[str],
    test_size: float,
    random_state: int,
    min_count_in_each_split: int = 1,
) -> tuple[list[str], list[str]]:
    if len(class_dates) <= 1:
        return sorted(class_dates), []

    if len(class_dates) == 2:
        return [class_dates[0]], [class_dates[1]]

    if min_count_in_each_split > 0 and len(class_dates) >= 2 * min_count_in_each_split:
        train_dates, test_dates = train_test_split(
            class_dates,
            test_size=test_size,
            random_state=random_state,
        )
        return sorted(train_dates), sorted(test_dates)

    # Fallback: preserve at least one example per side when possible without breaking validity.
    train_dates, test_dates = train_test_split(
        class_dates,
        test_size=max(0.5, min(test_size, 0.5)),
        random_state=random_state,
    )
    return sorted(train_dates), sorted(test_dates)


def generate_scene_split(
    dataset_root: Path,
    test_size: float = 0.25,
    random_state: int = 42,
    dates: list[str] | None = None,
    max_scenes: int | None = None,
    min_positive_scenes: int = 1,
    min_negative_scenes: int = 1,
) -> tuple[list[str], list[str]]:
    dataset_root = dataset_root.expanduser().resolve()
    date_list = dates or discover_dates(dataset_root)
    if not date_list:
        raise ValueError(f"No scene dates were found under {dataset_root}")

    if max_scenes is not None:
        if max_scenes <= 0:
            raise ValueError("max_scenes must be greater than 0.")
        if max_scenes < len(date_list):
            rng = np.random.default_rng(random_state)
            labels = np.array([1 if scene_has_fire(dataset_root, date_str) else 0 for date_str in date_list], dtype=int)
            class_dates = {0: [], 1: []}
            for date_str, label in zip(date_list, labels):
                class_dates[int(label)].append(date_str)

            if len(class_dates[0]) and len(class_dates[1]):
                if max_scenes < 2:
                    raise ValueError("max_scenes must be at least 2 when both fire/no-fire classes are present.")
                selected = []
                selected.extend(rng.choice(class_dates[0], size=min(len(class_dates[0]), max(1, min_negative_scenes)), replace=False).tolist())
                selected.extend(rng.choice(class_dates[1], size=min(len(class_dates[1]), max(1, min_positive_scenes)), replace=False).tolist())
                remaining = max_scenes - len(selected)
                if remaining > 0:
                    for label in [0, 1]:
                        remaining_dates = [d for d in class_dates[label] if d not in selected]
                        if remaining <= 0:
                            break
                        take = min(len(remaining_dates), remaining)
                        if take > 0:
                            selected.extend(rng.choice(remaining_dates, size=take, replace=False).tolist())
                            remaining -= take
                    if remaining > 0:
                        pool = [d for d in date_list if d not in selected]
                        if pool:
                            selected.extend(rng.choice(pool, size=min(len(pool), remaining), replace=False).tolist())
                date_list = sorted(selected)
            else:
                date_list = rng.choice(date_list, size=max_scenes, replace=False).tolist()
        else:
            date_list = list(date_list)

    if len(date_list) < 2:
        raise ValueError("At least 2 scenes are required to generate a train/test split.")

    labels = np.array([1 if scene_has_fire(dataset_root, date_str) else 0 for date_str in date_list], dtype=int)
    class_dates = {0: [], 1: []}
    for date_str, label in zip(date_list, labels):
        class_dates[int(label)].append(date_str)

    if len(class_dates[0]) and len(class_dates[1]):
        pos_train, pos_test = split_scene_class(
            class_dates[1],
            test_size=test_size,
            random_state=random_state,
            min_count_in_each_split=min_positive_scenes,
        )
        neg_train, neg_test = split_scene_class(
            class_dates[0],
            test_size=test_size,
            random_state=random_state,
            min_count_in_each_split=min_negative_scenes,
        )

        train_dates = sorted(pos_train + neg_train)
        test_dates = sorted(pos_test + neg_test)
        if not train_dates or not test_dates:
            raise ValueError("The selected scenes do not produce a valid train/test split.")
        return train_dates, test_dates

    train_dates, test_dates = train_test_split(
        date_list,
        test_size=test_size,
        random_state=random_state,
    )
    return sorted(train_dates), sorted(test_dates)


def build_split_id(train_dates: list[str], test_dates: list[str], test_size: float, random_state: int) -> str:
    seed = "|".join(sorted(train_dates) + sorted(test_dates))
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    test_label = str(test_size).replace(".", "p")
    return f"scene_split_{len(train_dates) + len(test_dates)}scenes_{test_label}test_rs{random_state}_{digest}"


def write_split_bundle(
    output_dir: Path,
    train_dates: list[str],
    test_dates: list[str],
    *,
    split_id: str | None = None,
    dataset_root: Path | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
    max_scenes: int | None = None,
    min_positive_scenes: int = 1,
    min_negative_scenes: int = 1,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_split_id = split_id or build_split_id(train_dates, test_dates, test_size, random_state)
    safe_split_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe_split_id).strip("_")
    if not safe_split_id:
        raise ValueError("split_id must contain at least one alphanumeric character.")

    train_path = output_dir / f"{safe_split_id}_train_dates.csv"
    test_path = output_dir / f"{safe_split_id}_test_dates.csv"
    metadata_path = output_dir / f"{safe_split_id}_metadata.json"

    pd.DataFrame({"date": sorted(train_dates)}).to_csv(train_path, index=False)
    pd.DataFrame({"date": sorted(test_dates)}).to_csv(test_path, index=False)

    metadata = {
        "split_id": safe_split_id,
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "train_file": str(train_path.name),
        "test_file": str(test_path.name),
        "train_dates": sorted(train_dates),
        "test_dates": sorted(test_dates),
        "n_train": len(train_dates),
        "n_test": len(test_dates),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "max_scenes": int(max_scenes) if max_scenes is not None else None,
        "min_positive_scenes": int(min_positive_scenes),
        "min_negative_scenes": int(min_negative_scenes),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return train_path, test_path, metadata_path


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    train_dates, test_dates = generate_scene_split(
        dataset_root,
        test_size=args.test_size,
        random_state=args.random_state,
        dates=args.dates,
        max_scenes=args.max_scenes,
        min_positive_scenes=args.min_positive_scenes,
        min_negative_scenes=args.min_negative_scenes,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path, test_path, metadata_path = write_split_bundle(
        output_dir,
        train_dates,
        test_dates,
        split_id=args.split_id,
        dataset_root=dataset_root,
        test_size=args.test_size,
        random_state=args.random_state,
        max_scenes=args.max_scenes,
        min_positive_scenes=args.min_positive_scenes,
        min_negative_scenes=args.min_negative_scenes,
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Split ID: {metadata_path.stem.replace('_metadata', '')}")
    print(f"Scenes used: {len(train_dates) + len(test_dates)}")
    print(f"Train scenes: {len(train_dates)}")
    print(f"Test scenes: {len(test_dates)}")
    print(f"Train file: {train_path.name}")
    print(f"Test file: {test_path.name}")
    print(f"Metadata file: {metadata_path.name}")
    print(f"Saved split bundle to {output_dir}")


if __name__ == "__main__":
    main()
