"""
sync_hf.py — Upload the local dataset folder to Hugging Face.

Usage:
  .venv/bin/python obtencion_imagenes/sync_hf.py

Reads HF_TOKEN (or huggingface_token) from the .env file in the project root.
Edit HF_REPO below to match your Hugging Face username/dataset-name.
"""

import os
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO      = "valentina2323/GERIS-Goes19-uruguay-fires"
DATASET_DIR  = Path(__file__).parent / "dataset"
ENV_FILE     = Path(__file__).parent.parent / ".env"
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
        print("❌ No se encontró el token de Hugging Face.")
        print("   Agregá 'HF_TOKEN=hf_...' al archivo .env en la raíz del proyecto.")
        sys.exit(1)

    if HF_REPO.startswith("your-username"):
        print("❌ Editá HF_REPO en sync_hf.py con tu usuario y nombre de repositorio.")
        sys.exit(1)

    if not DATASET_DIR.exists():
        print(f"❌ No existe la carpeta de dataset: {DATASET_DIR}")
        print("   Primero corré pipeline.py para descargar datos.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ huggingface_hub no está instalado. Corré: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # Create repo if it doesn't exist yet
    try:
        api.create_repo(repo_id=HF_REPO, repo_type="dataset", token=token, exist_ok=True, private=True)
        print(f"📦 Repositorio listo: https://huggingface.co/datasets/{HF_REPO}")
    except Exception as e:
        print(f"❌ Error al crear el repositorio: {e}")
        sys.exit(1)

    print(f"⬆️  Subiendo {DATASET_DIR} → {HF_REPO} ...")
    try:
        api.upload_folder(
            folder_path=str(DATASET_DIR),
            repo_id=HF_REPO,
            repo_type="dataset",
            token=token,
            commit_message="sync dataset",
        )
        print(f"\n✅ Dataset sincronizado correctamente.")
        print(f"   Ver en: https://huggingface.co/datasets/{HF_REPO}")
    except Exception as e:
        print(f"❌ Error durante la subida: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
