"""
pipeline.py — CLI entry point for the GOES dataset pipeline.

Usage:
  python pipeline.py download --region uruguay --start "2025-09-01 00:00" --end "2025-09-30 23:00" --products ABI-L1b-Rad-B07 ABI-L1b-Rad-B14 ABI-L2-FDCF-Mask ABI-L2-FDCF-DQF ABI-L1b-Rad-B13 ABI-L1b-Rad-B15 ABI-L1b-Rad-B02
  python pipeline.py status   --region uruguay --products ABI-L1b-Rad-B07 ABI-L1b-Rad-B14 ABI-L2-FDCF-Mask ABI-L2-FDCF-DQF ABI-L1b-Rad-B13 ABI-L1b-Rad-B15 ABI-L1b-Rad-B02
  python pipeline.py list-products
  python pipeline.py list-regions
"""

import argparse
import yaml
import os
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import downloader
import manifest

from dotenv import load_dotenv
load_dotenv()   # lee .env de la raíz del repo si existe; no falla si no existe


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_manifest(output_root):
    path = os.path.join(output_root, "manifest.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            try:
                result = future.result()
                results.append(result)
                icon = {"downloaded": "💾", "exists": "✅", "empty": "⚠️", "error": "❌"}.get(result["status"], "?")
                
                if result["status"] == "error":
                    print(f"  ❌ {result['timestamp']}  {result['product']:<30}  {result.get('substatus', result['status'])}")
                    print(f"     👉 Detalle real: {result.get('error')}")
                else:
                    print(f"  {icon} {result['timestamp']}  {result['product']:<30}  {result['status']}")
            
            except Exception as e:
                # Esto atrapa errores inesperados del downloader
                print(f"  ❌ Error crítico en descarga: {e}")

    # 1. Obtenemos los IDs de los productos configurados para verificar integridad
    all_product_ids = [p["id"] for p in products] 
    
    # 2. Obtenemos la configuración geográfica de la región para calcular coordenadas
    region_cfg = cfg["regions"][args.region] 
    
    # 3. Llamamos al nuevo manifest con los 4 argumentos requeridos
    manifest.update(output_root, results, all_product_ids, region_cfg)


    downloaded = sum(1 for r in results if r["status"] == "downloaded")
    skipped    = sum(1 for r in results if r["status"] == "exists")
    errors     = sum(1 for r in results if r["status"] in ("error", "empty"))
    print(f"\n✔ Descargados: {downloaded}  |  Ya existían: {skipped}  |  Errores: {errors}")
    print(f"📋 Manifest actualizado en: {output_root}/manifest.json")


def cmd_status(args, cfg):
    output_root = os.path.join(cfg["output_root"], args.region)
    required    = args.products if args.products else None
    manifest.status_report(output_root, required_products=required)
    manifest.export_metadata_csv(output_root)


def cmd_list_products(cfg):
    print("\n📡 Productos disponibles en config.yaml:\n")
    for p in cfg["products"]:
        band = f"B{p['band']:02d}" if p.get("band") else "  -"
        print(f"  {p['id']:<30}  {band}  {p['description']}")


def cmd_visualize(args, cfg):
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

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

    # Obtener el extent geográfico de la región desde config.yaml
    # Esto le dice a cartopy dónde está el array en el mapa del mundo
    region_cfg = cfg["regions"][args.region]
    lon_min = region_cfg["lon_min"]
    lon_max = region_cfg["lon_max"]
    lat_min = region_cfg["lat_min"]
    lat_max = region_cfg["lat_max"]
    extent  = [lon_min, lon_max, lat_min, lat_max]

    # Proyección PlateCarree: lat/lon directo, sin distorsión — la más simple
    proj = ccrs.PlateCarree()

    n_panels = 2 if mask is not None else 1
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(7 * n_panels, 6),
        subplot_kw={"projection": proj},  # todos los ejes usan la misma proyección
    )
    if n_panels == 1:
        axes = [axes]  # normalizar a lista para iterar igual en ambos casos

    fig.suptitle(f"GOES-19 — {args.region} — {ts}", fontsize=14)

    def add_map_features(ax):
        """Agrega costas, fronteras y grilla de lat/lon a un eje de cartopy."""
        # Limitar la vista al extent de la región
        ax.set_extent(extent, crs=proj)

        # Costas con resolución de 10m (la más detallada disponible en Natural Earth)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8, color="white")

        # Fronteras nacionales
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6, color="white", linestyle="--")

        # Grilla de lat/lon con etiquetas
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="white", alpha=0.5, linestyle=":")
        gl.top_labels   = False  # solo etiquetas abajo y a la izquierda
        gl.right_labels = False

    # ── Panel 1: Banda 7 (infrarrojo térmico) ───────────────────────────────
    # imshow con transform=proj y extent le dice a cartopy que el array cubre
    # exactamente el bounding box de la región
    im1 = axes[0].imshow(
        np.log1p(rad),
        origin="upper",       # fila 0 = norte (convención de imágenes satelitales)
        extent=extent,
        transform=proj,
        cmap="magma",
    )
    axes[0].set_title("Banda 7 — Infrarrojo (firma térmica)")
    add_map_features(axes[0])
    plt.colorbar(im1, ax=axes[0], label="log(1 + Rad)", shrink=0.7)

    # ── Panel 2: Máscara de fuego ────────────────────────────────────────────
    if mask is not None:
        mask_int = mask.astype(np.int8)
        # Fuego = DQF==0 (buena calidad), pero solo si no es un archivo todo-ceros
        if np.all(mask_int == 0):
            print("⚠️  Este timestamp tiene una máscara inválida (toda la imagen es DQF=0, región no procesada)")
            fuego = np.zeros_like(mask_int, dtype=float)
        else:
            fuego = (mask_int == 0).astype(float)
        im2 = axes[1].imshow(
            fuego,
            origin="upper",
            extent=extent,
            transform=proj,
            cmap="Reds",
            vmin=0, vmax=1,
        )
        axes[1].set_title(f"Máscara de fuego — {int(fuego.sum())} píxeles detectados")
        add_map_features(axes[1])
        plt.colorbar(im2, ax=axes[1], shrink=0.7)

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

    con_fuego, sin_fuego, invalidos, detalles = [], [], [], []
    for f in archivos:
        mask = np.load(os.path.join(mask_folder, f)).astype(np.int8)
        # Detectar archivos inválidos: toda la imagen es DQF=0 (región no procesada)
        # Esto pasa con timestamps anteriores a la operación del satélite
        if np.all(mask == 0):
            invalidos.append(f)
        else:
            # DQF=0 → fuego de buena calidad
            fire_pixels = int(np.sum(mask == 0))
            if fire_pixels > 0:
                con_fuego.append(f)
                detalles.append((f.replace(".npy", ""), fire_pixels))
            else:
                sin_fuego.append(f)

    total_validos = len(con_fuego) + len(sin_fuego)
    total = len(archivos)
    print(f"\n🔥 ESTADÍSTICAS DE FUEGO — {args.region}")
    print(f"{'='*45}")
    print(f"  Total archivos              : {total}")
    if invalidos:
        print(f"  ⚠️  Inválidos (no procesados): {len(invalidos)}")
    print(f"  Timestamps válidos          : {total_validos}")
    print(f"  Con fuego detectado         : {len(con_fuego)}  ({100*len(con_fuego)/total_validos:.1f}%)" if total_validos else "")
    print(f"  Sin fuego                   : {len(sin_fuego)}  ({100*len(sin_fuego)/total_validos:.1f}%)" if total_validos else "")

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


def cmd_retry(args, cfg):
    region = cfg["regions"].get(args.region)
    if not region:
        print(f"❌ Región '{args.region}' no encontrada en config.yaml")
        return

    output_root = os.path.join(cfg["output_root"], args.region)
    manifest_entries = load_manifest(output_root)
    if not manifest_entries:
        print(f"❌ No se encontró manifest.json o está vacío en: {output_root}")
        return

    product_map = {p["id"]: p for p in cfg["products"]}
    retry_entries = [
        e for e in manifest_entries
        if e["status"] in ("error", "empty")
        and (not args.products or e["product"] in args.products)
    ]

    if not retry_entries:
        print("✅ No hay descargas con error pendientes de reintento.")
        return

    if args.products:
        unknown = set(args.products) - set(product_map)
        if unknown:
            print(f"⚠️  Productos desconocidos ignorados: {', '.join(unknown)}")

    tasks = []
    for entry in retry_entries:
        prod = product_map.get(entry["product"])
        if not prod:
            print(f"⚠️  Producto no definido en config.yaml: {entry['product']}")
            continue

        # Convertimos el string "20250901_1200" a un objeto datetime
        ts_obj = datetime.strptime(entry["timestamp"], "%Y%m%d_%H%M")
        tasks.append((ts_obj, prod))
    
    if not tasks:
        print("❌ No hay tareas válidas para reintentar.")
        return

    max_workers = args.workers or cfg["max_workers"]
    print(f"🔁 Reintentando {len(tasks)} descargas con {max_workers} workers...\n")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                downloader.download_and_save, ts, prod, region, cfg["satellite"], cfg["domain"], output_root
            ): (ts, prod)
            for ts, prod in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            icon = {"downloaded": "💾", "exists": "✅", "empty": "⚠️", "error": "❌"}.get(result["status"], "?")
            print(f"  {icon} {result['timestamp']}  {result['product']:<30}  {result['status']}")

    # 1. Obtenemos todos los IDs definidos en el config.yaml para esta región
    all_ids = [p["id"] for p in cfg["products"]]
    
    # 2. Obtenemos la configuración de la región
    region_cfg = cfg["regions"][args.region]
    
    # 3. Actualizamos el manifest
    manifest.update(output_root, results, all_ids, region_cfg)


    downloaded = sum(1 for r in results if r["status"] == "downloaded")
    skipped    = sum(1 for r in results if r["status"] == "exists")
    errors     = sum(1 for r in results if r["status"] in ("error", "empty"))
    print(f"\n✔ Reintentos completados: descargados={downloaded}  |  ya existían={skipped}  |  errores={errors}")
    print(f"📋 Manifest actualizado en: {output_root}/manifest.json")

def cmd_download_camel(args, cfg):
    region_cfg = cfg["regions"].get(args.region)
    if not region_cfg:
        print(f"❌ Región '{args.region}' no encontrada en config.yaml")
        return

    output_root = os.path.join(cfg["output_root"], args.region)
    camel_dir   = region_cfg.get(
        "camel_emissivity_dir",
        os.path.join(output_root, "camel_emissivity"),
    )

    try:
        path = downloader.download_camel_climatology(args.month, camel_dir)
        print(f"✅ Climatología CAMEL V3 lista para el mes {args.month:02d}: {path}")
    except Exception as e:
        print(f"❌ Error descargando climatología CAMEL: {e}")

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

    # retry
    rt = sub.add_parser("retry", help="Retry downloads that failed previously")
    rt.add_argument("--region",   required=True, help="Region key from config.yaml")
    rt.add_argument("--products", nargs="+", help="Limit retry to these product IDs")
    rt.add_argument("--workers",  type=int, default=None, help="Parallel workers (overrides config)")

    # download-camel
    dc = sub.add_parser("download-camel", help="Descarga climatología de emisividad CAMEL V3 (LP DAAC)")
    dc.add_argument("--region", required=True, help="Region key from config.yaml")
    dc.add_argument("--month",  required=True, type=int, choices=range(1, 13), help="Mes calendario (1-12)")

    sp = sub.add_parser("spatial-report", help="Muestra el fuego por departamento para un timestamp")
    sp.add_argument("--region", required=True, help="Región (ej: uruguay)")
    sp.add_argument("--timestamp", required=True, help="Timestamp (ej: 20250926_1900)")

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
    elif args.command == "retry":
        cmd_retry(args, cfg)
    elif args.command == "download-camel":
        cmd_download_camel(args, cfg)
    elif args.command == "spatial-report":
        cmd_spatial_report(args, cfg)
    elif args.command == "visualize":     
        cmd_visualize(args, cfg)
    else:
        parser.print_help()
    

def cmd_spatial_report(args, cfg):
    import numpy as np
    output_root = os.path.join(cfg["output_root"], args.region)
    ts = args.timestamp
    mask_path = os.path.join(output_root, "ABI-L2-FDCF", f"{ts}.npy")

    if not os.path.exists(mask_path):
        print(f"❌ No se encontró la máscara para {ts}")
        return

    mask = np.load(mask_path)
    region_cfg = cfg["regions"][args.region]
    
    # Esta función ahora está integrada en tu manifest.py
    # Simplemente consultamos el manifest ya actualizado
    data = manifest.load(output_root)
    ts_data = data.get(ts, {})
    report = ts_data.get("fire", {}).get("spatial_report", {})

    print(f"\n📍 REPORTE ESPACIAL — {args.region.upper()} — {ts}")
    print("="*45)
    if not report:
        print("✅ No se detectaron focos de fuego en este timestamp.")
    else:
        for depto, count in sorted(report.items(), key=lambda x: x[1], reverse=True):
            print(f"  {depto:<20} {count:>5} píxeles")

if __name__ == "__main__":
    main()
