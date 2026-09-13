"""
One-off experiment for R2, step 2: run Part I + Part II on the padded
domain (extended land_mask for background, ROI clipped to the real
Uruguay polygon) and compare the resulting fire mask against both the
original (cropped) run and the NOAA reference mask.

Does NOT modify part1.py / part2.py / fdca_adapter.py. Uses
load_fdca_input() as-is, but overrides the padded input's region_mask
with the real Uruguay country polygon computed on the padded lat/lon
grid (build_region_mask(..., region_name="uruguay")) -- otherwise
build_surface_masks would fall back to the "uruguay_padded" bounding
box from config.yaml, since that name has no match in Natural Earth.
"""

import json
import os
import numpy as np

from fdca.dataset import default_dataset_root
from fdca.fdca_adapter import load_fdca_input, build_region_mask
from fdca.part1 import run_part1
from fdca.part2 import run_part2
from fdca.algorithm import _to_epoch
from fdca.metrics import evaluate, print_metrics

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="R2 padded-vs-original audit")
    parser.add_argument("--timestamp", default=None, help='single timestamp, e.g. "20251202_1500"')
    parser.add_argument("--timestamps", default=None,
                         help='comma-separated list, e.g. "20250926_1900,20260214_1500"')
    return parser.parse_args()

REGION_ORIGINAL = "uruguay"
REGION_PADDED = "uruguay_padded"
DATASET_ROOT = default_dataset_root()
CONFIG_PATH = "fdca/config.yaml"
FIRE_CODES_LOCAL = (10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35)

def load_geometry_xy(base: str) -> tuple[np.ndarray, np.ndarray]:
    with open(os.path.join(base, "geometry.json")) as f:
        geom = json.load(f)
    return np.array(geom["x"]), np.array(geom["y"])


def find_grid_offset(x_orig, y_orig, x_pad, y_pad, tol: float = 1e-6) -> tuple[int, int]:
    """
    Locate the (row, col) of the original grid's top-left corner inside the
    padded grid. Both crops sample the same fixed ABI x/y grid, so this is
    an exact index lookup, not a geographic reprojection.
    """
    col_offset = int(np.argmin(np.abs(x_pad - x_orig[0])))
    row_offset = int(np.argmin(np.abs(y_pad - y_orig[0])))

    if (abs(x_pad[col_offset] - x_orig[0]) > tol
            or abs(y_pad[row_offset] - y_orig[0]) > tol):
        raise ValueError(
            "Padded grid does not share the original grid's fixed x/y "
            "sampling -- offset lookup is not exact. Check both crops "
            "were downloaded from the same GOES product/resolution."
        )
    return row_offset, col_offset


def run_part1_part2(inp):
    """Run Part I + Part II with prev_fire_mask disabled (fixed-scene audit)."""
    fire_mask_p1, fail_char, candidates = run_part1(
        bt7=inp.bt7, rad7=inp.rad7, bt14=inp.bt14, rad14=inp.rad14,
        bt13=inp.bt13, rad13=inp.rad13, bt15=inp.bt15, refl2=inp.refl2,
        latitudes=inp.latitudes, longitudes=inp.longitudes,
        sza=inp.sza, glint_angle=inp.glint_angle, lza=inp.lza, azimuth=inp.azimuth,
        tpw=inp.tpw, emiss7=inp.emiss7, emiss14=inp.emiss14,
        lut_tpw=inp.lut_tpw, FPT=inp.FPT,
        coeffs7=inp.coeffs7, coeffs14=inp.coeffs14, coeffs13=inp.coeffs13,
        land_mask=inp.land_mask, region_mask=inp.region_mask,
        eco_mask=inp.eco_mask, data_quality=inp.data_quality,
    )
    fire_mask_p2, fail_char_p2, confirmed = run_part2(
        candidates=candidates,
        fire_mask=fire_mask_p1.copy(),
        fail_char_arr=fail_char.copy(),
        prev_fire_mask=None,
        current_epoch=_to_epoch(inp.scan_time),
    )
    return fire_mask_p1, fire_mask_p2, candidates, confirmed


def build_candidate_mask(shape, candidates, row_off: int = 0, col_off: int = 0):
    mask = np.zeros(shape, dtype=bool)
    L, W = shape
    for c in candidates:
        ci, cj = c.i - row_off, c.j - col_off
        if 0 <= ci < L and 0 <= cj < W:
            mask[ci, cj] = True
    return mask

def _add_counts(total: dict | None, m: dict) -> dict:
    """Micro-sum tp/fp/fn/tn across scenes for one metrics sub-block (e.g. m['part2'])."""
    keys = ("tp", "fp", "fn", "tn")
    if total is None:
        return {k: m[k] for k in keys}
    return {k: total[k] + m[k] for k in keys}


def _finish_counts(counts: dict) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(counts, precision=precision, recall=recall, f1=f1)


def run_one(timestamp: str) -> dict:
    """Run the R2 padded-vs-original comparison for one timestamp.
    Returns a dict with both metrics blocks and the pixel diff count,
    instead of only printing, so multiple timestamps can be aggregated.
    """
    global TIMESTAMP
    TIMESTAMP = timestamp
    print("=" * 70)
    print(f"Running Part I + II on ORIGINAL grid ({REGION_ORIGINAL}) | {timestamp}")
    print("=" * 70)
    inp_orig = load_fdca_input(
        timestamp=timestamp, region=REGION_ORIGINAL,
        dataset_root=DATASET_ROOT, config_path=CONFIG_PATH, verbose=True,
    )
    fire_mask_p1_orig, fire_mask_p2_orig, cand_orig, conf_orig = run_part1_part2(inp_orig)

    print("\n" + "=" * 70)
    print(f"Running Part I + II on PADDED grid ({REGION_PADDED}) | {timestamp}")
    print("=" * 70)
    inp_pad = load_fdca_input(
        timestamp=timestamp, region=REGION_PADDED,
        dataset_root=DATASET_ROOT, config_path=CONFIG_PATH, verbose=True,
    )
    base_padded = os.path.join(DATASET_ROOT, REGION_PADDED)
    inp_pad.region_mask = build_region_mask(
        inp_pad.latitudes, inp_pad.longitudes,
        region_name="uruguay", base_path=base_padded,
    )
    fire_mask_p1_pad, fire_mask_p2_pad, cand_pad, conf_pad = run_part1_part2(inp_pad)

    base_orig = os.path.join(DATASET_ROOT, REGION_ORIGINAL)
    x_orig, y_orig = load_geometry_xy(base_orig)
    x_pad, y_pad = load_geometry_xy(base_padded)
    row_off, col_off = find_grid_offset(x_orig, y_orig, x_pad, y_pad)

    L, W = fire_mask_p1_orig.shape
    sl = np.s_[row_off:row_off+L, col_off:col_off+W]
    fire_mask_p1_pad_c = fire_mask_p1_pad[sl]
    fire_mask_p2_pad_c = fire_mask_p2_pad[sl]

    candidate_mask_orig = build_candidate_mask((L, W), cand_orig)
    candidate_mask_pad_c = build_candidate_mask((L, W), cand_pad, row_off, col_off)

    reference_path = os.path.join(base_orig, "ABI-L2-FDCF-Mask", f"{timestamp}.npy")
    reference_mask = np.load(reference_path).astype(np.uint8)
    reference_masked = np.where(inp_orig.region_mask, reference_mask, 0)

    metrics_orig = evaluate(reference_masked, fire_mask_p1_orig, candidate_mask_orig, fire_mask_p2_orig)
    metrics_pad = evaluate(reference_masked, fire_mask_p1_pad_c, candidate_mask_pad_c, fire_mask_p2_pad_c)
    print_metrics(metrics_orig, reference_path)
    print_metrics(metrics_pad, reference_path)

    diff_mask = fire_mask_p2_orig != fire_mask_p2_pad_c
    n_diff = int(diff_mask.sum())
    print(f"\n{n_diff} pixels differ between original and padded runs "
          f"({100*n_diff/fire_mask_p2_orig.size:.3f}% of the scene)")

    return {
        "timestamp": timestamp,
        "metrics_orig": metrics_orig,
        "metrics_pad": metrics_pad,
        "n_diff": n_diff,
        "n_reference_fire": int(np.count_nonzero(np.isin(reference_masked, FIRE_CODES_LOCAL))),
    }

def main():
    args = parse_args()
    if args.timestamps:
        timestamps = [t.strip() for t in args.timestamps.split(",") if t.strip()]
    elif args.timestamp:
        timestamps = [args.timestamp]
    else:
        raise SystemExit("Pass --timestamp or --timestamps")

    results = [run_one(ts) for ts in timestamps]

    if len(results) > 1:
        total_orig = total_pad = None
        for r in results:
            total_orig = _add_counts(total_orig, r["metrics_orig"]["part2"])
            total_pad = _add_counts(total_pad, r["metrics_pad"]["part2"])

        print("\n" + "=" * 70)
        print(f"AGGREGATED OVER {len(results)} SCENES")
        print("=" * 70)
        s_orig, s_pad = _finish_counts(total_orig), _finish_counts(total_pad)
        print(f"ORIGINAL : precision={s_orig['precision']:.2%} recall={s_orig['recall']:.2%} "
              f"f1={s_orig['f1']:.2%} TP={s_orig['tp']} FP={s_orig['fp']} FN={s_orig['fn']}")
        print(f"PADDED   : precision={s_pad['precision']:.2%} recall={s_pad['recall']:.2%} "
              f"f1={s_pad['f1']:.2%} TP={s_pad['tp']} FP={s_pad['fp']} FN={s_pad['fn']}")
        print("\nPer-scene n_diff:")
        for r in results:
            print(f"  {r['timestamp']}: {r['n_diff']} pixels differ "
                  f"(n_reference_fire={r['n_reference_fire']})")


if __name__ == "__main__":
    main()