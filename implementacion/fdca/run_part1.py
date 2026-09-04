"""Command-line runner for FDCA Part I."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import DEFAULT_HF_REPO_ID, default_dataset_root, ensure_timestamp_data
from .fdca_adapter import load_fdca_input
from .part1 import run_part1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FDCA Part I on one GOES-19 scene"
    )
    parser.add_argument("--timestamp", required=True, help="Scene, e.g. 20251117_1820")
    parser.add_argument("--region", default="uruguay")
    parser.add_argument("--dataset-root", default=default_dataset_root(),
        help="Dataset root folder (default: $GERIS_DATASET_ROOT env var if set, else './dataset')")
    parser.add_argument("--config", default="fdca/config.yaml")
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Do not auto-download missing scenes from Hugging Face (fail instead)",
    )
    parser.set_defaults(download=True)
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default="results/part1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_timestamp_data(
        timestamp=args.timestamp,
        region=args.region,
        dataset_root=args.dataset_root,
        download=args.download,
        repo_id=args.repo_id,
    )

    print(f"FDCA Part I | region={args.region} | timestamp={args.timestamp}")
    inp = load_fdca_input(
        timestamp=args.timestamp,
        region=args.region,
        dataset_root=args.dataset_root,
        config_path=args.config,
        verbose=True,
    )

    fire_mask, fail_char, candidates = run_part1(
        bt7=inp.bt7,
        rad7=inp.rad7,
        bt14=inp.bt14,
        rad14=inp.rad14,
        bt13=inp.bt13,
        rad13=inp.rad13,
        bt15=inp.bt15,
        refl2=inp.refl2,
        latitudes=inp.latitudes,
        longitudes=inp.longitudes,
        sza=inp.sza,
        glint_angle=inp.glint_angle,
        lza=inp.lza,
        azimuth=inp.azimuth,
        tpw=inp.tpw,
        emiss7=inp.emiss7,
        emiss14=inp.emiss14,
        lut_tpw=inp.lut_tpw,
        FPT=inp.FPT,
        coeffs7=inp.coeffs7,
        coeffs14=inp.coeffs14,
        coeffs13=inp.coeffs13,
        land_mask=inp.land_mask,
        region_mask=getattr(inp, "region_mask", np.ones_like(inp.land_mask, dtype=bool)),
        eco_mask=inp.eco_mask,
        data_quality=inp.data_quality,
    )

    output_dir = (Path(args.output_dir) / args.timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "fire_mask_part1.npy", fire_mask)
    np.save(output_dir / "fail_char_part1.npy", fail_char)
    np.save(
        output_dir / "candidates_part1.npy",
        np.array([(candidate.i, candidate.j) for candidate in candidates], dtype=np.int32),
    )
    summary = {
        "timestamp": args.timestamp,
        "region": args.region,
        "shape": list(fire_mask.shape),
        "n_candidates": len(candidates),
        "n_fire_pixels": int(np.count_nonzero(fire_mask)),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Part I complete: {len(candidates):,} candidates")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
