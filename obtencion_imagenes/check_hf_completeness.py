"""
check_hf_completeness.py — Audita el repo de Hugging Face (GERIS-Goes19-uruguay-fires)
contra la grilla teórica de timestamps cada 10 minutos, para un rango de fechas dado,
y reporta qué falta por producto/banda.

No corre ningún tool del paquete fdca: sólo lista archivos remotos con
huggingface_hub.list_repo_files() y compara contra la grilla esperada. Pensado para
correr ANTES de pipeline.py download, así el --start/--end que le pases sale de acá
en vez de adivinar.

Por qué hace falta un script aparte (y no simplemente mirar manifest.json):
  - manifest.json vive en el filesystem LOCAL de quien lo generó; no refleja qué
    quedó efectivamente subido a HF (pueden divergir si algo falló en sync_hf.py
    o si otra persona subió desde otra máquina).
  - sync_hf.py rutea cada archivo por-timestamp a una carpeta con sufijo -YYYYMM
    (ver SHARDABLE_FOLDERS ahí) para no pasarse del límite de 10k archivos/carpeta
    de HF. Este script tiene que deshacer ese sharding para poder comparar correctamente.

Uso
---

  # sólo un par de productos, rango custom
  python check_hf_completeness.py --products ABI-L1b-Rad-B07 ABI-L1b-Rad-B14 \
      --start "2025-12-01 00:00" --end "2025-12-15 23:50"

  # incluir el sidecar _planck.json de las bandas IR (7/13/14/15)
  python check_hf_completeness.py --check-planck

Salidas en <output-dir>/ (default: results/hf_completeness/<run_id>/)
  report.md          resumen legible + comandos sugeridos de pipeline.py download
  missing.csv        una fila por (producto, timestamp faltante)
  summary.json        conteos y % de completitud por producto, en JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_REPO_ID = "valentina2323/GERIS-Goes19-uruguay-fires"
DEFAULT_REGION = "uruguay"

# Mismo default que pipeline.py cmd_download usa en su ejemplo de --help, y el
# mismo set de productos "por escena" (muestreados cada 10 min) del pipeline.
DEFAULT_PRODUCTS = [
    "ABI-L1b-Rad-B07",
    "ABI-L1b-Rad-B14",
    "ABI-L2-FDCF-Mask",
    "ABI-L2-FDCF-DQF",
    "ABI-L1b-Rad-B13",
    "ABI-L1b-Rad-B15",
    "ABI-L1b-Rad-B02",
]

# Bandas IR que tienen sidecar de coeficientes Planck (ver downloader.py PLANCK_BANDS).
PLANCK_PRODUCTS = {
    "ABI-L1b-Rad-B07",
    "ABI-L1b-Rad-B13",
    "ABI-L1b-Rad-B14",
    "ABI-L1b-Rad-B15",
}

# Carpetas que sync_hf.py particiona por mes (ver SHARDABLE_FOLDERS en sync_hf.py).
# Debe coincidir exactamente con esa lista o el des-sharding de abajo queda mal.
SHARDABLE_FOLDERS = {
    "ABI-L1b-Rad-B02", "ABI-L1b-Rad-B07", "ABI-L1b-Rad-B07-DFQ",
    "ABI-L1b-Rad-B13", "ABI-L1b-Rad-B14", "ABI-L1b-Rad-B15",
    "ABI-L2-FDCF-DQF", "ABI-L2-FDCF-Mask",
}

_MONTH_SUFFIX_RE = re.compile(r"^(.+)-(\d{6})$")
_TS_RE = re.compile(r"^(\d{8}_\d{4})")


# ── Auth (mismo patrón que sync_hf.py) ───────────────────────────────────────
def load_token(env_file: Path) -> str | None:
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip().lower()
            value = value.strip().strip('"').strip("'")
            if key in ("hf_token", "huggingface_token"):
                return value
    return os.environ.get("HF_TOKEN")


# ── Grilla teórica de timestamps ─────────────────────────────────────────────
def expected_timestamps(start: datetime, end: datetime, interval_min: int) -> list[str]:
    out, t = [], start
    delta = timedelta(minutes=interval_min)
    while t <= end:
        out.append(t.strftime("%Y%m%d_%H%M"))
        t += delta
    return out


# ── Listado remoto y des-sharding ────────────────────────────────────────────
def list_region_files(repo_id: str, region: str, token: str | None) -> list[str]:
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    prefix = f"{region}/"
    return [f for f in all_files if f.startswith(prefix)]


def build_found_index(
    files: list[str], region: str, products: list[str], check_planck: bool
) -> dict[str, dict[str, set]]:
    """
    Devuelve {producto: {"npy": {timestamps...}, "planck": {timestamps...}}}.

    Deshace el sharding por mes (carpeta "Producto-YYYYMM") comparando contra
    SHARDABLE_FOLDERS, igual que _flatten_sharded_downloads en dataset.py.
    """
    wanted = set(products)
    found: dict[str, dict[str, set]] = {p: {"npy": set(), "planck": set()} for p in products}
    prefix_len = len(region) + 1  # "uruguay/"

    for path in files:
        rel = path[prefix_len:]
        parts = rel.split("/", 1)
        if len(parts) != 2:
            continue
        folder, filename = parts

        base_product = folder
        match = _MONTH_SUFFIX_RE.match(folder)
        if match and match.group(1) in SHARDABLE_FOLDERS:
            base_product = match.group(1)

        if base_product not in wanted:
            continue

        ts_match = _TS_RE.match(filename)
        if not ts_match:
            continue
        ts = ts_match.group(1)

        if filename.endswith("_planck.json") and check_planck:
            found[base_product]["planck"].add(ts)
        elif filename == f"{ts}.npy":
            found[base_product]["npy"].add(ts)
        elif filename == f"{ts}_dqf.npy":
            # Sidecar DQF que puede vivir junto a la banda (legacy) o en su
            # propia carpeta ABI-L1b-Rad-B07-DFQ; no lo contamos como el
            # archivo principal del producto.
            continue

    return found


# ── Rangos contiguos para sugerir comandos de descarga ───────────────────────
def group_contiguous(missing_ts: list[str], interval_min: int) -> list[tuple[str, str, int]]:
    if not missing_ts:
        return []
    dts = sorted(datetime.strptime(t, "%Y%m%d_%H%M") for t in missing_ts)
    step = timedelta(minutes=interval_min)
    ranges = []
    range_start = range_end = dts[0]
    count = 1
    for dt in dts[1:]:
        if dt - range_end == step:
            range_end = dt
            count += 1
        else:
            ranges.append((range_start, range_end, count))
            range_start = range_end = dt
            count = 1
    ranges.append((range_start, range_end, count))
    return [(s.strftime("%Y-%m-%d %H:%M"), e.strftime("%Y-%m-%d %H:%M"), c) for s, e, c in ranges]


# ── Reporte ───────────────────────────────────────────────────────────────────
def build_report(
    args, missing_by_product: dict[str, list[str]], total_expected: int,
) -> str:
    lines = []
    add = lines.append
    add("# Auditoría de completitud — GOES-19 en Hugging Face\n")
    add(f"- Repo: `{args.repo_id}` (dataset)")
    add(f"- Región: `{args.region}`")
    add(f"- Rango: `{args.start}` → `{args.end}`  (cada {args.interval} min)")
    add(f"- Timestamps esperados por producto: {total_expected}")
    add(f"- Productos chequeados: {', '.join(args.products)}\n")

    add("## Resumen por producto\n")
    add("| producto | encontrados | faltantes | % completo |")
    add("|---|---|---|---|")
    for product in args.products:
        n_missing = len(missing_by_product.get(product, []))
        n_found = total_expected - n_missing
        pct = 100 * n_found / total_expected if total_expected else 0.0
        add(f"| {product} | {n_found} | {n_missing} | {pct:.1f}% |")
    add("")

    # Agrupar productos que comparten exactamente el mismo set de faltantes,
    # para sugerir un solo comando de pipeline.py con varios --products.
    by_missing_set: dict[frozenset, list[str]] = defaultdict(list)
    for product in args.products:
        key = frozenset(missing_by_product.get(product, []))
        if key:
            by_missing_set[key].append(product)

    add("## Rangos faltantes y comandos sugeridos\n")
    if not by_missing_set:
        add("No falta nada en el rango chequeado. ✅\n")
    for missing_set, products in by_missing_set.items():
        add(f"### {', '.join(products)}\n")
        ranges = group_contiguous(sorted(missing_set), args.interval)
        add(f"{len(missing_set)} timestamps faltantes en {len(ranges)} tramo(s) contiguo(s):\n")
        add("| desde | hasta | cantidad |")
        add("|---|---|---|")
        for start_s, end_s, count in ranges:
            add(f"| {start_s} | {end_s} | {count} |")
        add("")
        add("Comando(s) sugerido(s) (idempotente: `download_and_save` ya salta lo que "
            "existe localmente, así que no pasa nada si algún timestamp del tramo ya "
            "lo tenés):\n")
        add("```bash")
        for start_s, end_s, _ in ranges:
            add(
                f'python pipeline.py download --region {args.region} '
                f'--start "{start_s}" --end "{end_s}" '
                f'--products {" ".join(products)} --interval {args.interval}'
            )
        add("```")
        add("")

    if args.check_planck:
        add("## Sidecars `_planck.json` faltantes (bandas IR)\n")
        any_missing_planck = False
        for product in args.products:
            if product not in PLANCK_PRODUCTS:
                continue
            missing_planck = missing_by_product.get(f"{product}::planck", [])
            if missing_planck:
                any_missing_planck = True
                add(f"- `{product}`: {len(missing_planck)} timestamps sin "
                    f"`_planck.json` (el .npy principal puede estar presente igual)")
        if not any_missing_planck:
            add("Ningún faltante detectado.")
        add("")

    add("## Después de descargar\n")
    add("Correr `python obtencion_imagenes/sync_hf.py` para subir lo nuevo — ya se "
        "encarga de saltear lo que ya está en HF (`list_repo_files` + diff), así que "
        "es seguro correrlo aunque el dataset local tenga de más.\n")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Chequea huecos en el dataset GOES subido a Hugging Face",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--start", default="2025-11-01 00:00",
                    help='Inicio del rango, formato "YYYY-MM-DD HH:MM"')
    p.add_argument("--end", default="2026-01-31 23:50",
                    help='Fin del rango (inclusive), formato "YYYY-MM-DD HH:MM"')
    p.add_argument("--interval", type=int, default=10,
                    help="Cadencia de muestreo en minutos")
    p.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS,
                    help="Productos/bandas a chequear (nombres de carpeta en HF)")
    p.add_argument("--check-planck", action="store_true",
                    help="También chequear el sidecar _planck.json en bandas IR (7/13/14/15)")
    p.add_argument("--output-dir", default="results/hf_completeness")
    p.add_argument("--run-id", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    env_file = Path(__file__).resolve().parent.parent / ".env"
    token = load_token(env_file)

    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    end = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
    expected = expected_timestamps(start, end, args.interval)
    expected_set = set(expected)

    print("=" * 72)
    print(f"Chequeando {args.repo_id} | región={args.region}")
    print(f"Rango: {args.start} -> {args.end}  (cada {args.interval} min, "
          f"{len(expected)} timestamps esperados)")
    print(f"Productos: {', '.join(args.products)}")
    print("=" * 72)

    print("\nListando archivos remotos (puede tardar unos segundos)...")
    files = list_region_files(args.repo_id, args.region, token)
    print(f"  {len(files)} archivos bajo '{args.region}/' en el repo")

    found = build_found_index(files, args.region, args.products, args.check_planck)

    missing_by_product: dict[str, list[str]] = {}
    for product in args.products:
        found_ts = found[product]["npy"]
        missing = sorted(expected_set - found_ts)
        missing_by_product[product] = missing
        pct = 100 * (len(expected) - len(missing)) / len(expected) if expected else 0.0
        print(f"  {product:<22}: {len(expected) - len(missing):>5}/{len(expected)} "
              f"({pct:5.1f}%)  faltan {len(missing)}")

        if args.check_planck and product in PLANCK_PRODUCTS:
            found_planck = found[product]["planck"]
            missing_planck = sorted(expected_set - found_planck)
            missing_by_product[f"{product}::planck"] = missing_planck
            print(f"    └─ _planck.json: faltan {len(missing_planck)}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # missing.csv
    with open(out_dir / "missing.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["product", "timestamp"])
        for product in args.products:
            for ts in missing_by_product.get(product, []):
                writer.writerow([product, ts])

    # summary.json
    summary = {
        "repo_id": args.repo_id,
        "region": args.region,
        "start": args.start,
        "end": args.end,
        "interval_minutes": args.interval,
        "n_expected": len(expected),
        "products": {
            product: {
                "n_found": len(expected) - len(missing_by_product.get(product, [])),
                "n_missing": len(missing_by_product.get(product, [])),
                "pct_complete": round(
                    100 * (len(expected) - len(missing_by_product.get(product, [])))
                    / len(expected), 2,
                ) if expected else None,
                "missing": missing_by_product.get(product, []),
            }
            for product in args.products
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    report = build_report(args, missing_by_product, len(expected))
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"\nInforme  : {out_dir / 'report.md'}")
    print(f"CSV      : {out_dir / 'missing.csv'}")
    print(f"JSON     : {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
