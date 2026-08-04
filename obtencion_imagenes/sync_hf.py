"""
sync_hf.py — Uploads the local dataset to Hugging Face with a clean structure.

HF Structure:
  uruguay/
  ├── geometry.json                    ← GOES coordinates for the region
  ├── manifest.json                    ← download log
  ├── camel_emissivity/                ← CAMEL emissivity (if present)
  │   └── *.nc
  ├── ABI-L1b-Rad-B07/                ← IR band
  │   ├── units.json                   ← band metadata (1 per band)
  │   ├── 20250901_0000.npy
  │   ├── 20250901_0000_planck.json    ← Planck coefficients (1 per timestamp)
  │   ├── 20250901_0100.npy
  │   ├── 20250901_0100_planck.json
  │   └── ...
  ├── ABI-L1b-Rad-B14/
  │   ├── units.json
  │   ├── *.npy + *_planck.json
  │   └── ...
  └── ABI-L2-FDCF/                    ← derived product (no planck)
      ├── units.json
      ├── 20250901_0000.npy
      └── ...

Only uploads: .npy, .json, .nc
Excludes:     metadata.csv and any file not matching those extensions.
Automatically adds geometry.json, units.json, and camel_emissivity/ from
implementacion/ if they are not in dataset/.

Usage:
  python obtencion_imagenes/sync_hf.py
"""

import os
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO         = "valentina2323/GERIS-Goes19-uruguay-fires"
DATASET_DIR     = Path(__file__).parent / "dataset"
IMPL_DATA_DIR   = Path(__file__).parent.parent / "implementacion" / "data" / "uruguay"
ENV_FILE        = Path(__file__).parent.parent / ".env"

VALID_EXTENSIONS = {".npy", ".json", ".nc"}
EXCLUDE_FILES    = {"metadata.csv"}
# ─────────────────────────────────────────────────────────────────────────────


def load_token():
    """Read token from .env file."""
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key   = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key in ("hf_token", "huggingface_token"):
            return value
    return None


def collect_dataset_files(dataset_dir: Path):
    """Iterates through dataset/ and collects valid files with relative paths."""
    files = []
    for root, _, filenames in os.walk(dataset_dir):
        for fname in filenames:
            if fname in EXCLUDE_FILES:
                continue
            if Path(fname).suffix.lower() not in VALID_EXTENSIONS:
                continue
            local_path = Path(root) / fname
            rel_path = str(local_path.relative_to(dataset_dir))
            files.append((local_path, rel_path))
    return files


def add_extras_from_implementacion(files: list, dataset_dir: Path):
    """
    Adds geometry.json, units.json per band, and camel_emissivity/
    from implementacion/data/uruguay/ if they are not already in dataset/.
    """
    existing_paths = {rel for _, rel in files}

    # ── geometry.json region level ─────────────────────────────────────────
    geom_rel = "uruguay/geometry.json"
    if geom_rel not in existing_paths:
        geom_src = IMPL_DATA_DIR / "geometry.json"
        if geom_src.exists():
            files.append((geom_src, geom_rel))
            print(f"  + geometry.json (from implementation)")

    # ── units.json per band ─────────────────────────────────────────────────
    # The downloader generates a single units.json per band folder.
    # If missing (e.g., data downloaded with an older version), it pulls it from implementacion.
    bands = set()
    for _, rel in files:
        parts = Path(rel).parts
        if len(parts) >= 2 and parts[1].startswith("ABI-"):
            bands.add(parts[1])

    for band in sorted(bands):
        units_rel = f"uruguay/{band}/units.json"
        if units_rel not in existing_paths:
            # Look for units.json or *_units.json in implementacion for that band
            band_dir = IMPL_DATA_DIR / band
            if band_dir.exists():
                # First search for units.json directly
                direct = band_dir / "units.json"
                if direct.exists():
                    files.append((direct, units_rel))
                    print(f"  + {band}/units.json (from implementation)")
                else:
                    # Fallback: use any *_units.json as a template
                    units_files = list(band_dir.glob("*_units.json"))
                    if units_files:
                        files.append((units_files[0], units_rel))
                        print(f"  + {band}/units.json (from implementation)")

    # ── camel_emissivity ─────────────────────────────────────────────────────
    camel_in_dataset = any("camel_emissivity" in rel for _, rel in files)
    if not camel_in_dataset:
        camel_dir = IMPL_DATA_DIR / "camel_emissivity"
        if camel_dir.exists():
            for f in sorted(camel_dir.iterdir()):
                if f.suffix.lower() in VALID_EXTENSIONS:
                    rel = f"uruguay/camel_emissivity/{f.name}"
                    files.append((f, rel))
                    print(f"  + camel_emissivity/{f.name} (from implementation)")

    return files


def main():
    token = load_token() or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Hugging Face not found.")
        print("   Add 'HF_TOKEN=hf_...' to the .env file in the project root.")
        sys.exit(1)

    if not DATASET_DIR.exists():
        print(f"❌ Dataset folder does not exist: {DATASET_DIR}")
        print("   First, run pipeline.py to download the data.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, CommitOperationAdd
    except ImportError:
        print("❌ huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    # ── Collect files ──────────────────────────────────────────────────
    print(f"📂 Scanning: {DATASET_DIR}")
    files = collect_dataset_files(DATASET_DIR)
    files = add_extras_from_implementacion(files, DATASET_DIR)

    # ── Summary ──────────────────────────────────────────────────────────────
    npy_count = sum(1 for _, r in files if r.endswith(".npy"))
    json_count = sum(1 for _, r in files if r.endswith(".json"))
    nc_count = sum(1 for _, r in files if r.endswith(".nc"))
    planck_count = sum(1 for _, r in files if r.endswith("_planck.json"))
    units_count = sum(1 for _, r in files if r.endswith("_units.json"))

    bands = {}
    for _, rel in files:
        parts = Path(rel).parts
        if len(parts) >= 2 and parts[1].startswith("ABI-"):
            bands[parts[1]] = bands.get(parts[1], 0) + 1

    print(f"\n📊 Files to upload: {len(files)}")
    print(f"   .npy: {npy_count}  |  .json: {json_count} (planck: {planck_count}, units: {units_count})  |  .nc: {nc_count}")
    if bands:
        print(f"   Bands:")
        for band, count in sorted(bands.items()):
            print(f"     {band}: {count} files")
    print()

    # ── Subir ────────────────────────────────────────────────────────────────
    api = HfApi()

    try:
        api.create_repo(repo_id=HF_REPO, repo_type="dataset", token=token,
                        exist_ok=True, private=True)
        print(f"📦 Repository: https://huggingface.co/datasets/{HF_REPO}")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        sys.exit(1)

    print(f"⬆️  Uploading {len(files)} files...")
    try:
        operations = [
            CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(local))
            for local, rel in files
        ]

        api.create_commit(
            repo_id=HF_REPO,
            repo_type="dataset",
            token=token,
            operations=operations,
            commit_message="sync dataset",
        )
        print(f"\n✅ Dataset was succesfully synchronized.")
        print(f"   Ver en: https://huggingface.co/datasets/{HF_REPO}")
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
