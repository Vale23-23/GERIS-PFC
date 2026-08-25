"""Dataset helpers for reproducible FDCA scene downloads."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_HF_REPO_ID = "valentina2323/GERIS-Goes19-uruguay-fires"


def missing_required_files(timestamp: str, region: str, dataset_root: str | Path) -> list[Path]:
    """Return the files required to construct an FDCA input."""
    base = Path(dataset_root) / region
    required = [
        base / "ABI-L1b-Rad-B07" / f"{timestamp}.npy",
        base / "ABI-L1b-Rad-B14" / f"{timestamp}.npy",
        base / "ABI-L1b-Rad-B07" / f"{timestamp}_planck.json",
        base / "ABI-L1b-Rad-B14" / f"{timestamp}_planck.json",
        base / "geometry.json",
    ]
    return [path for path in required if not path.exists()]


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
