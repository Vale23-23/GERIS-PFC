"""Convert one-per-scene feature CSVs to compressed Parquet files.

The source CSV directory is never modified. Existing Parquet files are skipped
unless --force is supplied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convierte CSVs de features por escena a Parquet.")
    parser.add_argument("--input-dir", required=True, help="Directorio con archivos YYYYMMDD_HHMM.csv.")
    parser.add_argument("--output-dir", required=True, help="Directorio destino de los Parquet.")
    parser.add_argument("--compression", choices=("zstd", "snappy", "gzip", "none"), default="zstd")
    parser.add_argument("--force", action="store_true", help="Sobrescribe Parquet existentes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay CSVs en {input_dir}")

    compression = None if args.compression == "none" else args.compression
    converted = 0
    skipped = 0
    for source in files:
        target = output_dir / f"{source.stem}.parquet"
        if target.exists() and not args.force:
            skipped += 1
            continue
        frame = pd.read_csv(source)
        frame.to_parquet(target, index=False, compression=compression, engine="pyarrow")
        converted += 1
        print(f"{source.name} -> {target.name}: {len(frame)} filas")

    print(f"Convertidos: {converted} | omitidos: {skipped} | destino: {output_dir}")


if __name__ == "__main__":
    main()
