"""Dataset helpers for reproducible FDCA scene downloads."""

from __future__ import annotations
from datetime import datetime
import os
from pathlib import Path

from dotenv import load_dotenv
from .tpw_downloader import cycle_path_for_timestamp
DEFAULT_HF_REPO_ID = "valentina2323/GERIS-Goes19-uruguay-fires"

def default_dataset_root() -> str:
    """
    Resolve the dataset root independent of which branch checkout the user
    is standing in (e.g. .../obtencion_imagenes/ vs .../testing-fdca-real/).

    Follows the same per-machine .env pattern as GERIS_OUTPUT_ROOT /
    GERIS_GOES2GO_CACHE: if GERIS_DATASET_ROOT is set, both branches can
    point at one shared physical folder; otherwise falls back to the
    relative "dataset" folder for simple single-checkout setups.
    """
    load_dotenv()
    return os.getenv("GERIS_DATASET_ROOT", "dataset")

def missing_required_files(timestamp: str, region: str, dataset_root: str | Path) -> list[Path]:
    """Return the files required to construct an FDCA input.

    B02, B13 and B15 are intentionally excluded: fdca_adapter.py treats them
    as optional (refl2, the hybrid-longwave band, and the BT15 cloud test all
    degrade gracefully when they're absent).

    TPW-GFS and CAMEL V3 emissivity ARE required here even though
    fdca_adapter.py currently falls back to climatology/placeholder values
    when they're missing -- that fallback exists for algorithm development,
    not something this reproducible-dataset check should mask.
    """
    base = Path(dataset_root) / region
    month = timestamp[4:6]

    required = [
        base / "ABI-L1b-Rad-B07" / f"{timestamp}.npy",
        base / "ABI-L1b-Rad-B14" / f"{timestamp}.npy",
        base / "ABI-L1b-Rad-B07" / f"{timestamp}_planck.json",
        base / "ABI-L1b-Rad-B14" / f"{timestamp}_planck.json",
        base / "geometry.json",
        Path(cycle_path_for_timestamp(
            datetime.strptime(timestamp, "%Y%m%d_%H%M"), str(base)
        )),
    ]
    missing = [path for path in required if not path.exists()]

    # CAMEL V3 climatology filenames are month-dependent, not a fixed name
    # (e.g. CAM5K30EMCLIM_emis_climatology_07Month_V003.nc), so they need a
    # glob check instead of a plain Path.exists().
    camel_dir = base / "camel_emissivity"
    if not camel_dir.exists() or not list(camel_dir.glob(f"*{month}Month*.nc")):
        missing.append(camel_dir / f"*{month}Month*.nc")

    # The TPW LUT ships with the package (fdca/data/tpw_lut.csv), not with
    # the per-timestamp HF dataset. build_tpw_lut() has no fallback if it's
    # missing, so it's checked here too -- but note download_timestamp()
    # cannot fetch it (see ensure_timestamp_data below).
    # Both now live in the dataset itself (HF), not the package.
    tpw_lut_path = Path(dataset_root) / "tpw_lut.csv"
    if not tpw_lut_path.exists():
        missing.append(tpw_lut_path)

    eco_mask_path = base / "eco_mask.npy"
    if not eco_mask_path.exists():
        missing.append(eco_mask_path)

    return missing


def download_timestamp(
    timestamp: str,
    region: str,
    dataset_root: str | Path,
    repo_id: str = DEFAULT_HF_REPO_ID,
) -> None:
    """Download one scene and its ancillary files from Hugging Face."""
    load_dotenv()
    from huggingface_hub import snapshot_download

    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    month = timestamp[4:6]
    token = os.getenv("HF_TOKEN") or None
    # TPW-GFS files are now keyed by GFS cycle (00/06/12/18 UTC), not by ABI
    # timestamp — resolve the cycle this scene falls into before building the pattern.
    _scene_dt = datetime.strptime(timestamp, "%Y%m%d_%H%M")
    _cycle_hour = (_scene_dt.hour // 6) * 6
    _cycle_str = _scene_dt.replace(
        hour=_cycle_hour, minute=0, second=0, microsecond=0
    ).strftime("%Y%m%d_%H%M")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        allow_patterns=[
            f"{region}/*/{timestamp}.npy",
            f"{region}/*/{timestamp}_planck.json",
            f"{region}/*/{timestamp}_dqf.npy",
            f"{region}/geometry.json",
            f"{region}/camel_emissivity/*{month}Month*.nc",
            f"{region}/TPW-GFS/{_cycle_str}.npy",
            f"{region}/eco_mask.npy",
            "tpw_lut.csv",
        ],
        local_dir=str(dataset_root),
    )


def ensure_timestamp_data(
    timestamp: str,
    region: str,
    dataset_root: str | Path,
    download: bool = False,
    repo_id: str = DEFAULT_HF_REPO_ID,
) -> None:
    """Validate a scene, optionally downloading it when it is not local."""
    missing = missing_required_files(timestamp, region, dataset_root)
    if missing and download:
        print(f"Datos incompletos para {timestamp}; descargando desde {repo_id}...")
        download_timestamp(timestamp, region, dataset_root, repo_id)
        missing = missing_required_files(timestamp, region, dataset_root)

    if missing:
        tpw_lut_path = Path(__file__).resolve().parent / "data" / "tpw_lut.csv"
        missing_text = "\n".join(f"  - {path}" for path in missing)
        command = (
            "python -m fdca.run_part1 "
            f"--timestamp {timestamp} --region {region} "
            f"--dataset-root {dataset_root} --download"
        )

        raise FileNotFoundError(
            "Faltan archivos obligatorios para esta escena:\n"
            f"{missing_text}\n\n"
            "Ejecutá nuevamente con --download para obtenerlos desde Hugging Face:\n"
            f"  {command}"
        )
