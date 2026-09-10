"""
reset_hf_dataset.py — Deletes ALL contents of the dataset on Hugging Face.

After running this, use pipeline.py download to re-download
and sync_hf.py to upload with a clean structure.

Usage:
  python obtencion_imagenes/reset_hf_dataset.py
"""

import os
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO  = "valentina2323/GERIS-Goes19-uruguay-fires"
ENV_FILE = Path(__file__).parent.parent / ".env"
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


def main():
    token = load_token() or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Hugging Face token not found.")
        print("   Add 'HF_TOKEN=hf_...' to the .env file in the project root.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, CommitOperationDelete
    except ImportError:
        print("❌ huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=HF_REPO, repo_type="dataset", token=token,
                    exist_ok=True, private=True)

    # List current files
    print(f"🔍 Listing files in {HF_REPO} ...")
    repo_files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset", token=token)

    if not repo_files:
        print("✅ The repository is already empty. Nothing to delete.")
        return

    print(f"🗑️  Deleting {len(repo_files)} files...")
    operations = [CommitOperationDelete(path_in_repo=f) for f in repo_files]

    api.create_commit(
        repo_id=HF_REPO,
        repo_type="dataset",
        token=token,
        operations=operations,
        commit_message="🗑️ Full dataset reset",
    )

    print(f"✅ Clean repository: https://huggingface.co/datasets/{HF_REPO}")
    print()
    print("Next steps:")
    print("  1. python obtencion_imagenes/pipeline.py download --region uruguay ...")
    print("  2. python obtencion_imagenes/sync_hf.py")


if __name__ == "__main__":
    main()