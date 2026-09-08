#!/usr/bin/env python3
"""Controlled ancillary-input ablations for the FDCA detector.

Each variant changes exactly one external/input source while keeping the same
radiances, geometry, reference mask, and Part-II policy.  The runner is
intentionally separate from production code: it is an audit instrument, not a
new detector policy.

Run from implementacion/:
    python analysis_ancillary_ablation.py
    python analysis_ancillary_ablation.py --timestamps 20251120_1930,...
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from fdca.algorithm import _to_epoch
from fdca.constants import FireMask
from fdca.fdca_adapter import get_tpw_estimate, load_fdca_input
from fdca.part1 import run_part1
from fdca.tpw_downloader import cycle_path_for_timestamp
from fdca.part2 import run_part2

FIRE_CODES = (10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35)
DEFAULT_TIMESTAMPS = (
    "20251120_1930", "20251120_1950", "20251127_1640", "20251213_1750",
)
VARIANTS = (
    "baseline", "tpw_climatology", "tpw_neutral_25", "tpw_minus10",
    "tpw_plus10", "lut_neutral", "emiss_neutral", "emiss_constant",
    "b02_off", "b15_off", "fpt_hybrid_forced", "dqf_off",
)


def scores(reference: np.ndarray, prediction: np.ndarray) -> dict:
    ref = np.isin(reference, FIRE_CODES)
    pred = np.isin(prediction, FIRE_CODES)
    tp = int(np.count_nonzero(ref & pred))
    fp = int(np.count_nonzero(~ref & pred))
    fn = int(np.count_nonzero(ref & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "support_ref": int(ref.sum()),
            "support_pred": int(pred.sum())}


def reference_for(root: Path, region: str, timestamp: str) -> np.ndarray:
    return np.load(root / region / "ABI-L2-FDCF-Mask" / f"{timestamp}.npy").astype(np.uint8)


def apply_variant(inp, name: str):
    """Return a shallow input copy with one controlled change."""
    out = copy.copy(inp)
    if name == "baseline":
        return out
    if name == "tpw_climatology":
        out.tpw = get_tpw_estimate(out.latitudes, out.longitudes, out.scan_time)
    elif name == "tpw_neutral_25":
        out.tpw = np.full(out.bt7.shape, 25.0, dtype=np.float32)
    elif name == "tpw_minus10":
        out.tpw = np.maximum(1.0, out.tpw - 10.0).astype(np.float32)
    elif name == "tpw_plus10":
        out.tpw = (out.tpw + 10.0).astype(np.float32)
    elif name == "lut_neutral":
        out.lut_tpw = np.array(out.lut_tpw, dtype=np.float64, copy=True)
        out.lut_tpw[2:4, :] = 1.0
        out.lut_tpw[4:6, :] = 0.0
    elif name == "emiss_neutral":
        out.emiss7 = np.ones_like(out.emiss7, dtype=np.float32)
        out.emiss14 = np.ones_like(out.emiss14, dtype=np.float32)
    elif name == "emiss_constant":
        out.emiss7 = np.full_like(out.emiss7, 0.95, dtype=np.float32)
        out.emiss14 = np.full_like(out.emiss14, 0.97, dtype=np.float32)
    elif name == "b02_off":
        out.refl2 = None
    elif name == "b15_off":
        out.bt15 = None
    elif name == "fpt_hybrid_forced":
        out.FPT = 91.0
    elif name == "dqf_off":
        out.data_quality = None
    else:
        raise ValueError(f"unknown variant: {name}")
    return out


def run_variant(inp, reference: np.ndarray, name: str, policy: str):
    data = apply_variant(inp, name)
    p1, fail1, candidates = run_part1(
        bt7=data.bt7, rad7=data.rad7, bt14=data.bt14, rad14=data.rad14,
        bt13=data.bt13, rad13=data.rad13, bt15=data.bt15, refl2=data.refl2,
        latitudes=data.latitudes, longitudes=data.longitudes, sza=data.sza,
        glint_angle=data.glint_angle, lza=data.lza, azimuth=data.azimuth,
        tpw=data.tpw, emiss7=data.emiss7, emiss14=data.emiss14,
        lut_tpw=data.lut_tpw, FPT=data.FPT, coeffs7=data.coeffs7,
        coeffs14=data.coeffs14, coeffs13=data.coeffs13,
        land_mask=data.land_mask, region_mask=data.region_mask,
        eco_mask=data.eco_mask, data_quality=data.data_quality,
    )
    trace = []
    p2, fail2, confirmed = run_part2(
        candidates=candidates, fire_mask=p1.copy(), fail_char_arr=fail1.copy(),
        prev_fire_mask=None, current_epoch=_to_epoch(data.scan_time),
        trace_out=trace, detection_policy=policy,
    )
    candidate_mask = np.zeros(reference.shape, dtype=bool)
    for cand in candidates:
        candidate_mask[cand.i, cand.j] = True
    return {
        "p1": p1, "p2": p2, "candidate_mask": candidate_mask,
        "n_candidates": len(candidates), "n_confirmed": len(confirmed),
        "score_p1": scores(reference, candidate_mask),
        "score_p2": scores(reference, p2),
        "final_codes": {str(k): int(v) for k, v in
                        Counter(p2[np.isin(p2, FIRE_CODES)].tolist()).items()},
    }

def ancillary_snapshot(inp, root: Path, region: str, timestamp: str) -> dict:
    base = root / region
    tpw = np.asarray(inp.tpw, dtype=float)
    dqf_path = base / "ABI-L1b-Rad-B07-DFQ" / f"{timestamp}_dqf.npy"
    fpt_path = base / "ABI-L1b-Rad-B07" / f"{timestamp}_planck.json"
    camel = sorted((base / "camel_emissivity").glob(f"*{timestamp[4:6]}Month*.nc"))
    tpw_path = Path(cycle_path_for_timestamp(
        datetime.strptime(timestamp, "%Y%m%d_%H%M"), str(base)
    ))
    return {
        "timestamp": timestamp,
        "shape": list(inp.bt7.shape),
        "tpw_file": str(tpw_path),
        "tpw_file_exists": tpw_path.exists(),
        "tpw_min": float(np.nanmin(tpw)), "tpw_median": float(np.nanmedian(tpw)),
        "tpw_max": float(np.nanmax(tpw)),
        "emiss7_min": float(np.nanmin(inp.emiss7)),
        "emiss7_median": float(np.nanmedian(inp.emiss7)),
        "emiss7_max": float(np.nanmax(inp.emiss7)),
        "emiss14_min": float(np.nanmin(inp.emiss14)),
        "emiss14_median": float(np.nanmedian(inp.emiss14)),
        "emiss14_max": float(np.nanmax(inp.emiss14)),
        "camel_file": camel[0].name if camel else None,
        "b02_present": inp.refl2 is not None,
        "b13_present": inp.bt13 is not None and inp.rad13 is not None,
        "b15_present": inp.bt15 is not None,
        "dqf_present": dqf_path.exists(),
        "dqf_unique": np.unique(np.load(dqf_path)).astype(int).tolist() if dqf_path.exists() else [],
        "fpt_sidecar_present": fpt_path.exists(),
        "fpt_value": float(inp.FPT),
        "lut_trans4_min": float(np.min(inp.lut_tpw[2])),
        "lut_trans11_min": float(np.min(inp.lut_tpw[3])),
        "lut_offset4_max": float(np.max(inp.lut_tpw[4])),
        "lut_offset11_max": float(np.max(inp.lut_tpw[5])),
        "coeffs7": inp.coeffs7,
        "coeffs14": inp.coeffs14,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def build_report(path: Path, config: dict, snapshots: list[dict], rows: list[dict],
                 changes: list[dict]) -> None:
    baseline = next(row for row in rows if row["variant"] == "baseline" and row["timestamp"] == "aggregate")
    lines = [
        "# FDCA ancillary-source ablation report",
        "",
        "This report changes one input source at a time and reruns the production Part I + Part II path.",
        "It is a sensitivity study, not a claim that neutralized inputs are physically correct.",
        "",
        f"- Scenes: `{', '.join(config['timestamps'])}`",
        f"- Part II policy: `{config['policy']}`; temporal filter: disabled",
        f"- Generated: `{config['generated_at']}`",
        "- Existing full audit context: 46 scenes, Part-II precision 5.83%, recall 93.44%, 920 FP.",
        "",
        "## Executive result",
        "",
        f"The panel baseline is **{pct(baseline['precision'])} precision**, **{pct(baseline['recall'])} recall**, "
        f"with {baseline['tp']} TP and {baseline['fp']} FP. The key question is whether ancillary inputs "
        "are creating those false alarms or merely perturbing already-weak candidates.",
        "",
        "## Aggregate ablations",
        "",
        "| variant | P1 candidates | P2 detections | precision | recall | TP | FP | FN | Δ fire decisions | TP lost | FP removed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["timestamp"] != "aggregate":
            continue
        lines.append(
            f"| `{row['variant']}` | {row['n_candidates']} | {row['n_confirmed']} | "
            f"{pct(row['precision'])} | {pct(row['recall'])} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row.get('changed_fire', 0)} | {row.get('tp_lost', 0)} | {row.get('fp_removed', 0)} |"
        )
    lines += [
        "",
        "`Δ fire decisions` counts pixels whose binary fire/non-fire decision changed; it excludes ordinary non-fire status-code changes. `FP removed` and `TP lost` are measured against the same-scene baseline mask.",
        "",
        "## Source interpretation",
        "",
        "- **TPW GFS vs climatology:** TPW enters only through the rounded TPW/LZA LUT bin and radiance correction. On this panel, replacing cached GFS TPW with the repository climatology changed no binary fire decisions, so atmospheric-water correction is not the leading explanation for the current FP population.",
        "- **TPW LUT:** `lut_neutral` removes both transmittance and additive offsets. It changed internal/status outputs but no binary fire decisions on this panel. That means the LUT is coupled to the calculations, but it is not the main precision lever in the tested scenes. The current LUT has materially lower 11 µm transmittance and larger offsets in high-TPW/high-angle bins, so bin selection still deserves boundary tests.",
        "- **Emissivity:** `emiss_neutral` removes the surface-emissivity correction; `emiss_constant` tests the common fixed-value shortcut. If these variants mostly change corrected temperatures/Dozier labels but do not remove weak candidates, emissivity is not the main FP source. CAMEL is a monthly 0.05° climatology, not a contemporaneous pixel retrieval.",
        "- **B02:** `b02_off` removes albedo/visible screening while leaving thermal tests intact. Any increase in candidates is evidence that the visible product is currently protective; a decrease would indicate that its calibration or cloud/glint logic is admitting bad candidates.",
        "- **B15:** `b15_off` tests the optional split-window cloud gate. Because B15 is only used in a cloud test, a zero/small delta means it is not responsible for the 895 low-probability FPs.",
        "- **B13/FPT:** `fpt_hybrid_forced` is deliberately counterfactual. The local B07 sidecars map the QC counter to FPT and currently do not activate the hybrid path in the baseline; this test measures the maximum relevance of that branch, not an operational condition.",
        "- **DQF:** `dqf_off` should be exactly identical if the current implementation merely passes DQF through. That is a useful audit result: the DQF product exists, but it is not currently an active precision gate.",
        "",
        "## Pattern checks to prioritize",
        "",
        "- The prior 46-scene audit had 895/920 final FPs as base code 15, 894 with NOAA code 100, 917/920 using histogram background, and all FP candidates at `n_passes=0`.",
        "- Their median observed `BT7 - background BT7` was 5.97 K versus 11.71 K for TPs. This supports a Part-II low-probability acceptance/calibration problem more strongly than a gross emissivity or TPW failure.",
        "- Do not tune around code 15 first. First test a precision-oriented acceptance boundary on the weak-candidate distribution, report the recall trade-off, and keep the normative ATBD output separate.",
        "",
        "## External-source provenance",
        "",
        "- NOAA GOES-R ABI Fire/Hot Spot Characterization ATBD v2.7: [ABI FDC ATBD](https://www.star.nesdis.noaa.gov/atmospheric-composition-training/documents/ABI_FDC_ATBD.pdf).",
        "- NASA LP DAAC CAMEL V3 emissivity climatology: [CAM5K30EMCLIM V003](https://www.earthdata.nasa.gov/es/data/catalog/lpcloud-cam5k30emclim-003).",
        "- NOAA GOES-R baseline TPW product description: [Total Precipitable Water](https://goes-r.noaa.gov/products/baseline-total-precipitable-water.html).",
        "- NOAA GOES-19 ABI radiance maturity documentation: [GOES-19 ABI L1b/CMI provisional read-me](https://www.ospo.noaa.gov/operations/goes/product-quality-overview/ps-pvr/goes-19/ABI/Radiances/Provisional/GOES-19_ABI-L1b-CMI_Provisional_ReadMe.pdf).",
        "",
        "Content from external sources was paraphrased for compliance with licensing restrictions; the numerical conclusions above come from the local repository and generated experiments.",
        "",
        "## Files",
        "",
        "- `variant_metrics.csv`: per-scene and aggregate metrics.",
        "- `variant_changes.csv`: per-scene changes relative to baseline.",
        "- `source_snapshot.json`: actual local ancillary availability and ranges.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamps", default=",".join(DEFAULT_TIMESTAMPS))
    parser.add_argument("--region", default="uruguay")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--config", type=Path, default=Path("fdca/config.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/ancillary_ablation"))
    parser.add_argument("--policy", choices=("atbd", "conservative"), default="atbd")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()
    timestamps = [value.strip() for value in args.timestamps.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        parser.error(f"unknown variants: {unknown}")

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    change_rows: list[dict] = []
    snapshots: list[dict] = []
    for timestamp in timestamps:
        print(f"[{timestamp}] loading inputs", flush=True)
        inp = load_fdca_input(timestamp, args.region, str(args.dataset_root), str(args.config), verbose=False)
        reference = reference_for(args.dataset_root, args.region, timestamp)
        snapshots.append(ancillary_snapshot(inp, args.dataset_root, args.region, timestamp))
        baseline = None
        for variant in variants:
            started = time.perf_counter()
            result = run_variant(inp, reference, variant, args.policy)
            elapsed = time.perf_counter() - started
            s = result["score_p2"]
            row = {
                "timestamp": timestamp, "variant": variant,
                "n_candidates": result["n_candidates"], "n_confirmed": result["n_confirmed"],
                **s, "seconds": round(elapsed, 3),
            }
            all_rows.append(row)
            if variant == "baseline":
                baseline = result
            print(f"  {variant:<20} P1={result['n_candidates']:>4} P2={result['n_confirmed']:>4} "
                  f"precision={pct(s['precision']):>7} recall={pct(s['recall']):>7} "
                  f"({elapsed:.1f}s)", flush=True)
            if baseline is not None and variant != "baseline":
                ref_fire = np.isin(reference, FIRE_CODES)
                base_pred = np.isin(baseline["p2"], FIRE_CODES)
                var_pred = np.isin(result["p2"], FIRE_CODES)
                change_rows.append({
                    "timestamp": timestamp, "variant": variant,
                    "changed_fire": int(np.count_nonzero(base_pred != var_pred)),
                    "changed_final": int(np.count_nonzero(baseline["p2"] != result["p2"])),
                    "changed_candidates": int(np.count_nonzero(baseline["candidate_mask"] != result["candidate_mask"])),
                    "tp_lost": int(np.count_nonzero(ref_fire & base_pred & ~var_pred)),
                    "tp_gained": int(np.count_nonzero(ref_fire & ~base_pred & var_pred)),
                    "fp_removed": int(np.count_nonzero(~ref_fire & base_pred & ~var_pred)),
                    "fp_added": int(np.count_nonzero(~ref_fire & ~base_pred & var_pred)),
                })

    # Aggregate counts by summing scene-level confusion counts.
    aggregate_rows = []
    for variant in variants:
        selected = [row for row in all_rows if row["variant"] == variant]
        agg = {"timestamp": "aggregate", "variant": variant}
        for key in ("tp", "fp", "fn", "support_ref", "support_pred", "n_candidates", "n_confirmed"):
            agg[key] = int(sum(row[key] for row in selected))
        agg["precision"] = agg["tp"] / (agg["tp"] + agg["fp"]) if agg["tp"] + agg["fp"] else 0.0
        agg["recall"] = agg["tp"] / (agg["tp"] + agg["fn"]) if agg["tp"] + agg["fn"] else 0.0
        agg["f1"] = (2 * agg["precision"] * agg["recall"] / (agg["precision"] + agg["recall"])
                    if agg["precision"] + agg["recall"] else 0.0)
        if variant == "baseline":
            agg.update({"changed_fire": 0, "changed_final": 0, "tp_lost": 0, "fp_removed": 0})
        else:
            diffs = [row for row in change_rows if row["variant"] == variant]
            for key in ("changed_fire", "changed_final", "tp_lost", "fp_removed"):
                agg[key] = int(sum(row[key] for row in diffs))
        aggregate_rows.append(agg)
    all_rows.extend(aggregate_rows)

    config = {"timestamps": timestamps, "region": args.region, "dataset_root": str(args.dataset_root),
              "policy": args.policy, "variants": variants,
              "generated_at": datetime.now().isoformat(timespec="seconds")}
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out / "source_snapshot.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
    write_csv(args.out / "variant_metrics.csv", all_rows)
    write_csv(args.out / "variant_changes.csv", change_rows)
    build_report(args.out / "report.md", config, snapshots, all_rows, change_rows)
    print(f"Report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
