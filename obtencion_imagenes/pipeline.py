"""
pipeline.py — CLI entry point for the GOES dataset pipeline.

Usage:
  python pipeline.py download --region uruguay --start "2025-09-01 00:00" --end "2025-09-30 23:00" --products ABI-L1b-Rad-B07 ABI-L2-FDCF
  python pipeline.py status   --region uruguay --products ABI-L1b-Rad-B07 ABI-L2-FDCF
  python pipeline.py list-products
  python pipeline.py list-regions
"""

import argparse
import yaml
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import downloader
import manifest


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_timestamps(start_str, end_str, interval_hours=1):
    start = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
    end   = datetime.strptime(end_str,   "%Y-%m-%d %H:%M")
    delta = timedelta(hours=interval_hours)
    ts, t = [], start
    while t <= end:
        ts.append(t)
        t += delta
    return ts


def cmd_download(args, cfg):
    region = cfg["regions"].get(args.region)
    if not region:
        print(f"❌ Región '{args.region}' no encontrada en config.yaml")
        return

    product_ids = args.products
    products    = [p for p in cfg["products"] if p["id"] in product_ids]
    if not products:
        print("❌ Ningún producto válido encontrado. Usa 'list-products' para ver opciones.")
        return

    unknown = set(product_ids) - {p["id"] for p in products}
    if unknown:
        print(f"⚠️  Productos desconocidos ignorados: {', '.join(unknown)}")

    timestamps   = get_timestamps(args.start, args.end, args.interval)
    output_root  = os.path.join(cfg["output_root"], args.region)
    satellite    = cfg["satellite"]
    domain       = cfg["domain"]
    max_workers  = args.workers or cfg["max_workers"]

    tasks = [(ts, prod) for ts in timestamps for prod in products]
    print(f"🚀 Descargando {len(tasks)} archivos con {max_workers} workers...\n")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                downloader.download_and_save, ts, prod, region, satellite, domain, output_root
            ): (ts, prod)
            for ts, prod in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            icon = {"downloaded": "💾", "exists": "✅", "empty": "⚠️", "error": "❌"}.get(result["status"], "?")
            print(f"  {icon} {result['timestamp']}  {result['product']:<30}  {result['status']}")

    manifest.update(output_root, results)

    downloaded = sum(1 for r in results if r["status"] == "downloaded")
    skipped    = sum(1 for r in results if r["status"] == "exists")
    errors     = sum(1 for r in results if r["status"] in ("error", "empty"))
    print(f"\n✔ Descargados: {downloaded}  |  Ya existían: {skipped}  |  Errores: {errors}")
    print(f"📋 Manifest actualizado en: {output_root}/manifest.json")


def cmd_status(args, cfg):
    output_root = os.path.join(cfg["output_root"], args.region)
    required    = args.products if args.products else None
    manifest.status_report(output_root, required_products=required)


def cmd_list_products(cfg):
    print("\n📡 Productos disponibles en config.yaml:\n")
    for p in cfg["products"]:
        band = f"B{p['band']:02d}" if p.get("band") else "  -"
        print(f"  {p['id']:<30}  {band}  {p['description']}")


def cmd_visualize(args, cfg):
    import numpy as np
    import matplotlib.pyplot as plt

    output_root = os.path.join(cfg["output_root"], args.region)
    ts          = args.timestamp  # e.g. "20250901_1200"

    rad_path  = os.path.join(output_root, "ABI-L1b-Rad-B07", f"{ts}.npy")
    mask_path = os.path.join(output_root, "ABI-L2-FDCF",     f"{ts}.npy")

    if not os.path.exists(rad_path):
        print(f"❌ No se encontró la imagen de radiancia para {ts}")
        print(f"   Buscando en: {rad_path}")
        return

    rad  = np.load(rad_path)
    mask = np.load(mask_path) if os.path.exists(mask_path) else None

    fig, axes = plt.subplots(1, 2 if mask is not None else 1, figsize=(14, 6))
    fig.suptitle(f"GOES-19 — {args.region} — {ts}", fontsize=14)

    ax_rad = axes[0] if mask is not None else axes
    im1 = ax_rad.imshow(np.log1p(rad), cmap="magma")
    ax_rad.set_title("Banda 7 — Infrarrojo (firma térmica)")
    ax_rad.axis("off")
    plt.colorbar(im1, ax=ax_rad, label="log(1 + Rad)")

    if mask is not None:
        fuego = (mask == 0).astype(int)
        im2   = axes[1].imshow(fuego, cmap="Reds", vmin=0, vmax=1)
        axes[1].set_title(f"Máscara de fuego — {int(fuego.sum())} píxeles detectados")
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()


def cmd_fire_stats(args, cfg):
    import numpy as np
    output_root  = os.path.join(cfg["output_root"], args.region)
    mask_folder  = os.path.join(output_root, "ABI-L2-FDCF")

    if not os.path.exists(mask_folder):
        print(f"❌ No se encontró la carpeta de máscaras: {mask_folder}")
        print("   Asegurate de haber descargado el producto ABI-L2-FDCF.")
        return

    archivos = sorted([f for f in os.listdir(mask_folder) if f.endswith(".npy")])
    if not archivos:
        print("⚠️  No hay máscaras descargadas todavía.")
        return

    con_fuego, sin_fuego, detalles = [], [], []
    for f in archivos:
        mask = np.load(os.path.join(mask_folder, f))
        n    = int(np.sum(mask == 0))  # DQF=0 → high confidence fire
        if n > 0:
            con_fuego.append(f)
            detalles.append((f.replace(".npy", ""), n))
        else:
            sin_fuego.append(f)

    total = len(archivos)
    print(f"\n🔥 ESTADÍSTICAS DE FUEGO — {args.region}")
    print(f"{'='*45}")
    print(f"  Total timestamps analizados : {total}")
    print(f"  Con fuego detectado         : {len(con_fuego)}  ({100*len(con_fuego)/total:.1f}%)")
    print(f"  Sin fuego                   : {len(sin_fuego)}  ({100*len(sin_fuego)/total:.1f}%)")

    if detalles:
        detalles.sort(key=lambda x: x[1], reverse=True)
        print(f"\n🔝 Top 10 timestamps con más píxeles de fuego:")
        print(f"  {'Timestamp':<20} {'Píxeles fuego':>15}")
        print(f"  {'-'*35}")
        for ts, n in detalles[:10]:
            print(f"  {ts:<20} {n:>15,}")


def cmd_list_regions(cfg):
    print("\n🗺  Regiones disponibles en config.yaml:\n")
    for name, coords in cfg["regions"].items():
        print(f"  {name:<20}  lat [{coords['lat_min']}, {coords['lat_max']}]  "
              f"lon [{coords['lon_min']}, {coords['lon_max']}]")


def main():
    parser = argparse.ArgumentParser(description="GOES dataset pipeline")
    sub    = parser.add_subparsers(dest="command")

    # download
    dl = sub.add_parser("download", help="Download products for a date range")
    dl.add_argument("--region",   required=True,  help="Region key from config.yaml")
    dl.add_argument("--start",    required=True,  help='Start datetime "YYYY-MM-DD HH:MM"')
    dl.add_argument("--end",      required=True,  help='End datetime "YYYY-MM-DD HH:MM"')
    dl.add_argument("--products", required=True,  nargs="+", help="Product IDs from config.yaml")
    dl.add_argument("--interval", type=int, default=1, help="Interval in hours (default: 1)")
    dl.add_argument("--workers",  type=int, default=None, help="Parallel workers (overrides config)")

    # status
    st = sub.add_parser("status", help="Show manifest status for a region")
    st.add_argument("--region",   required=True, help="Region key from config.yaml")
    st.add_argument("--products", nargs="+",     help="Check completeness for these products")

    # list-products / list-regions
    sub.add_parser("list-products", help="List all products defined in config.yaml")
    sub.add_parser("list-regions",  help="List all regions defined in config.yaml")

    # visualize
    viz = sub.add_parser("visualize", help="Show Band 7 image and fire mask for a timestamp")
    viz.add_argument("--region",    required=True, help="Region key from config.yaml")
    viz.add_argument("--timestamp", required=True, help='Timestamp to visualize, e.g. "20250901_1200"')

    # fire-stats
    fs = sub.add_parser("fire-stats", help="Show fire detection statistics for a region")
    fs.add_argument("--region", required=True, help="Region key from config.yaml")

    args = parser.parse_args()
    cfg  = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

    if args.command == "download":
        cmd_download(args, cfg)
    elif args.command == "status":
        cmd_status(args, cfg)
    elif args.command == "list-products":
        cmd_list_products(cfg)
    elif args.command == "list-regions":
        cmd_list_regions(cfg)
    elif args.command == "fire-stats":
        cmd_fire_stats(args, cfg)
    elif args.command == "visualize":
        cmd_visualize(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
