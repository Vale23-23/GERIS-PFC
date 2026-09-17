"""
Genera, con UN SOLO COMANDO, una tabla de features por cada timestamp pedido,
apta para entrenar un modelo de ML. Por defecto escribe Parquet comprimido con
Zstandard; también puede escribir CSV o ambos formatos. Por adentro corre
fdca.run_audit, que calcula la física y el contexto de fondo.

Es incremental: si ya existe la salida del timestamp en --output-dir, no la
vuelve a generar (salvo --force). Así se pueden sumar escenas nuevas sin
recomputar las existentes.

Columnas descartadas a propósito (ver conversación): verdict, ref_code (se
guarda aparte, solo informativo), pred_code, ref_base_code, pred_base_code,
base_code, final_code, upgraded_code, stage*, fail_char_*, reached_candidate,
confirmed_part2, eliminated, elim_reason, detection_policy, policy_*,
conf_thr*, conf_bkg7_*, conf_bt7c_*, p2_*, mg_*, fire_frac, fire_temp, frp,
dozier_valid, bt7_min_thr, bt7_refl_thr, is_cloudy, sat_flag, in_roi.

Requiere correrse en un entorno donde el paquete `fdca` sea importable (este
script invoca `python -m fdca.run_audit` como subproceso).
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

ID_COLUMNS = ["timestamp", "i", "j"]

# Ya viene binarizada (0/1) usando FIRE_CODES = (10-15, 30-35) contra la
# mascara de referencia de NOAA -- ver run_audit.py linea 423.
TARGET_COLUMN = "ref_fire"

# Se conserva solo para inspeccion manual. NUNCA agregar a feature_sets.yaml.
INFO_ONLY_COLUMNS = ["ref_code"]

# Mediciones fisicas crudas + contexto de fondo. Ver conversacion para el
# detalle de por que se excluye cada columna del rastro de decision del
# algoritmo auditado (part1.py / part2.py).
SAFE_FEATURE_COLUMNS = [
    "bt7", "bt14_eff", "bt15",
    "albedo", "refl", "refl2",
    "tpw", "sza", "lza", "glint_angle",
    "emiss7", "emiss14", "eco_code",
    "diff_bt7_bt14",
    "bt7_bkg", "bt7_bkg_std",
    "bt14_bkg", "bt14_bkg_std",
    "albedo_bkg", "reflb", "std_reflb",
    "bt7_minus_bkg7", "bt14_minus_bkg14", "refl_minus_reflb",
    "n_passes", "rad_diff_sigma",
    "bt7_corr", "bt14_corr",
]

FINAL_COLUMNS = ID_COLUMNS + [TARGET_COLUMN] + INFO_ONLY_COLUMNS + SAFE_FEATURE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un CSV de features por timestamp (un solo comando, incremental)."
    )
    parser.add_argument(
        "--timestamps",
        default=None,
        help="Timestamps separados por coma (ej. 20251115_0850,20251116_1200). "
        "Si se omite, se procesan TODOS los timestamps disponibles en --dataset-root/--region "
        "(los que ya tengan CSV se saltean igual, salvo --force).",
    )
    parser.add_argument("--dataset-root", default=None, help="Se pasa a --dataset-root de run_audit.py. Si se omite, usa el default de fdca.")
    parser.add_argument("--region", default="uruguay", help="Se pasa a --region de run_audit.py.")
    parser.add_argument("--config", default=None, help="Se pasa a --config de run_audit.py, si se especifica.")
    parser.add_argument("--temporal-source", default="none", choices=("reference", "own", "none"), help="Se pasa a --temporal-source de run_audit.py.")
    parser.add_argument(
        "--all-pixels",
        action="store_true",
        help="Vuelca la escena entera en vez de solo los pixeles de interes que ya arma run_audit.py "
        "(fuegos reales + detecciones propias + candidatos). Sin esto (default) ya viene con la curaduria "
        "de negativos duros que pide model_architecture.md.",
    )
    parser.add_argument("--output-dir", required=True, help="Carpeta donde queda una tabla por timestamp.")
    parser.add_argument(
        "--output-format",
        choices=("parquet", "csv", "both"),
        default="parquet",
        help="Formato de salida. Default: parquet comprimido con zstd.",
    )
    parser.add_argument("--audit-raw-output-dir", default="results/audit", help="Carpeta donde run_audit.py escribe su corrida cruda (interno, no es el resultado final).")
    parser.add_argument("--no-download", action="store_true", help="Se pasa a --no-download de run_audit.py.")
    parser.add_argument("--force", action="store_true", help="Regenera igual las salidas que ya existan en el formato elegido.")
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime que haria, sin correr nada.")
    return parser.parse_args()


def build_run_id(timestamps: list[str]) -> str:
    digest = hashlib.sha1("|".join(sorted(timestamps)).encode("utf-8")).hexdigest()[:10]
    return f"build_features_{len(timestamps)}scenes_{digest}"


def run_audit(args: argparse.Namespace, timestamps: list[str], run_id: str) -> Path:
    cmd = [
        sys.executable, "-m", "fdca.run_audit",
        "--timestamps", ",".join(timestamps),
        "--region", args.region,
        "--temporal-source", args.temporal_source,
        "--output-dir", args.audit_raw_output_dir,
        "--run-id", run_id,
    ]
    if args.dataset_root:
        cmd += ["--dataset-root", args.dataset_root]
    if args.config:
        cmd += ["--config", args.config]
    if args.all_pixels:
        cmd.append("--all-pixels")
    if args.no_download:
        cmd.append("--no-download")

    print("Ejecutando:", " ".join(cmd))
    pixels_csv = Path(args.audit_raw_output_dir) / run_id / "pixels.csv"
    if args.dry_run:
        return pixels_csv

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"fdca.run_audit termino con codigo {result.returncode}. "
            "Revisa el output de arriba (probablemente falta algun archivo de la escena "
            "o el paquete fdca no esta en el PYTHONPATH)."
        )
    if not pixels_csv.exists():
        raise FileNotFoundError(f"run_audit.py no genero el archivo esperado: {pixels_csv}")
    return pixels_csv


def select_safe_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FINAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"El pixels.csv de run_audit.py no tiene estas columnas esperadas: {missing}. "
            "Puede que run_audit.py haya cambiado de formato -- revisar SAFE_FEATURE_COLUMNS en este script."
        )
    return df[FINAL_COLUMNS].copy()


def output_paths(output_dir: Path, timestamp: str, output_format: str) -> list[Path]:
    suffixes = {
        "parquet": [".parquet"],
        "csv": [".csv"],
        "both": [".parquet", ".csv"],
    }
    return [output_dir / f"{timestamp}{suffix}" for suffix in suffixes[output_format]]


def split_and_write_per_timestamp(
    clean_df: pd.DataFrame,
    output_dir: Path,
    output_format: str,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for timestamp, group in clean_df.groupby("timestamp"):
        if output_format in ("csv", "both"):
            group.to_csv(output_dir / f"{timestamp}.csv", index=False)
        if output_format in ("parquet", "both"):
            group.to_parquet(
                output_dir / f"{timestamp}.parquet",
                index=False,
                compression="zstd",
                engine="pyarrow",
            )
        written.append(timestamp)
    return sorted(written)


def resolve_requested_timestamps(args: argparse.Namespace) -> list[str]:
    if args.timestamps:
        return [t.strip() for t in args.timestamps.split(",") if t.strip()]

    if not args.dataset_root:
        raise ValueError(
            "Para autodescubrir timestamps hace falta --dataset-root explicito "
            "(no se puede inferir sin invocar el resto del paquete fdca)."
        )

    try:
        from fdca.run_audit import discover_timestamps
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar fdca.run_audit.discover_timestamps para autodescubrir timestamps. "
            "Corre este script desde el root del proyecto (donde el paquete `fdca` sea importable), "
            "o especifica --timestamps a mano."
        ) from exc

    available = discover_timestamps(
        dataset_root=args.dataset_root,
        region=args.region,
        download=not args.no_download,
    )
    if not available:
        raise SystemExit(f"No se encontraron timestamps completos en {args.dataset_root}/{args.region}")
    print(f"Autodescubiertos {len(available)} timestamps disponibles en {args.dataset_root}/{args.region}.")
    return available


def main() -> None:
    args = parse_args()
    requested = resolve_requested_timestamps(args)

    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.force:
        to_generate = requested
        already_have = []
    else:
        already_have = [
            timestamp for timestamp in requested
            if all(path.exists() for path in output_paths(output_dir, timestamp, args.output_format))
        ]
        to_generate = [t for t in requested if t not in already_have]

    if already_have:
        print(f"Ya existen ({len(already_have)}), se saltean: {already_have}")

    if not to_generate:
        print("Nada para generar. Usa --force si queres regenerar igual.")
        return

    run_id = build_run_id(to_generate)
    raw_pixels_csv = run_audit(args, to_generate, run_id)

    if args.dry_run:
        print(
            f"[dry-run] se hubiera leído {raw_pixels_csv} y escrito "
            f"{args.output_format} por timestamp en {output_dir}"
        )
        return

    df = pd.read_csv(raw_pixels_csv)
    clean_df = select_safe_columns(df)
    written = split_and_write_per_timestamp(clean_df, output_dir, args.output_format)

    missing_after = sorted(set(to_generate) - set(written))
    if missing_after:
        print(f"AVISO: estos timestamps se pidieron pero no aparecieron en pixels.csv (revisar si existen en el dataset): {missing_after}")

    print(f"\nEscribí {len(written)} tabla(s) en {output_dir} (formato={args.output_format}):")
    for timestamp in written:
        scene_df = clean_df[clean_df["timestamp"] == timestamp]
        n_fire = int(scene_df[TARGET_COLUMN].sum())
        outputs = ", ".join(path.name for path in output_paths(output_dir, timestamp, args.output_format))
        print(f"  {outputs} -> {len(scene_df)} filas, {n_fire} positivos")

    print(f"\nColumnas de feature ({len(SAFE_FEATURE_COLUMNS)}): {SAFE_FEATURE_COLUMNS}")
    print(f"Columna informativa (NO USAR como feature): {INFO_ONLY_COLUMNS}")
    print(f"Columna target: {TARGET_COLUMN}")


if __name__ == "__main__":
    main()