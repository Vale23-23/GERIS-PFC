"""
sync_hf_raw.py — Uploads raw GOES-16 .nc training data (known-fire month) to a separate Hugging Face dataset repo.

This is independent from sync_hf.py: these files are raw NOAA .nc downloads (GOES-16, ABI-L2-CMIPF), not the processed .npy/+metadata 
structure produced by downloader.py for the GERIS-PFC (GOES-19) pipeline. Kept in its own repo so it isn't mixed with the other 
operational dataset.

Usage:
  python obtencion_imagenes/sync_hf_training_raw.py --local-dir /path/to/goes_16
"""

import argparse
import os
from pathlib import Path

ENV_FILE = Path(__file__).parent.parent / ".env"
HF_REPO  = "valentina2323/GERIS-goes16-raw" 


def load_token():
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key in ("hf_token", "huggingface_token"):
            return value
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", required=True,
                         help="Path to the goes_16 folder (contains 202201/, 202112/, ...)")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        print(f"❌ Folder not found: {local_dir}")
        return

    token = load_token() or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found in .env or environment.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=HF_REPO, repo_type="dataset", token=token,
                     exist_ok=True, private=True)
    print(f"📦 Repository: https://huggingface.co/datasets/{HF_REPO}")

    print(f"⬆️  Uploading {local_dir} → {HF_REPO} ...")
    api.upload_folder(
        repo_id=HF_REPO,
        repo_type="dataset",
        folder_path=str(local_dir),
        allow_patterns=["*.nc"],   # only raw NetCDF files
        commit_message="Add GOES-16 raw fire training data",
    )
    print("✅ Upload complete.")
    print(f"   View at: https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()