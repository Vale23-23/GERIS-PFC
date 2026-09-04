#!/usr/bin/env python3
"""Plot the eco-mask codes 150–153 and the audit false negatives/positives.

Usage examples:
    python plot_audit_pixels.py --audit 20260903_211636
    python plot_audit_pixels.py --audit 0
    python plot_audit_pixels.py --audit latest
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from functools import lru_cache

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

try:
    from fdca.fdca_adapter import build_region_mask, compute_latlon_grid
except ModuleNotFoundError:  # ejecución directa desde la carpeta implementacion/
    from fdca_adapter import build_region_mask, compute_latlon_grid


BASE_DIR = Path(__file__).resolve().parent
AUDIT_ROOT = BASE_DIR / "results" / "audit"

# bbox real aprox. de Uruguay, para el fallback sin geopandas (fix #3:
# antes usaba ymin/ymax de axvspan como si fueran latitudes, cuando en
# realidad son fracción 0-1 del eje -> el rectángulo quedaba corrido).
URUGUAY_LON_RANGE = (-58.5, -53.1)
URUGUAY_LAT_RANGE = (-34.9, -30.1)


def list_audit_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def resolve_audit_dir(root: Path, audit_arg: str) -> Path:
    dirs = list_audit_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"No hay auditorías en {root}")

    if not audit_arg or audit_arg.lower() == "latest":
        return dirs[-1]

    # Numeric index among the sorted audit folders.
    if audit_arg.isdigit():
        idx = int(audit_arg)
        if 0 <= idx < len(dirs):
            return dirs[idx]
        raise IndexError(f"Índice fuera de rango: {audit_arg}. Hay {len(dirs)} auditorías.")

    # Direct folder name or relative path.
    candidate = Path(audit_arg)
    if candidate.exists():
        return candidate.resolve()

    for folder in dirs:
        if folder.name == audit_arg or folder.name.startswith(audit_arg):
            return folder

    raise FileNotFoundError(f"No encontré la auditoría '{audit_arg}' en {root}")


def read_pixels(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: v for k, v in row.items()})
    return rows


def parse_float(value: str | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def build_plot_data(rows: list[dict]):
    eco_points = []
    fn_points = []
    fp_points = []

    for row in rows:
        in_roi_raw = row.get("in_roi")
        if in_roi_raw is not None:
            try:
                if int(float(str(in_roi_raw).strip())) == 0:
                    continue
            except ValueError:
                pass

        lat = parse_float(row.get("lat"))
        lon = parse_float(row.get("lon"))
        if lat == lat and lon == lon:
            verdict = (row.get("verdict") or "").strip()
            eco_code = parse_int(row.get("eco_code"))

            if eco_code in {150, 151, 152, 153}:
                eco_points.append((lat, lon, eco_code))
            if verdict == "FN":
                fn_points.append((lat, lon))
            elif verdict == "FP":
                fp_points.append((lat, lon))

    return eco_points, fn_points, fp_points


@lru_cache(maxsize=1)
def _load_world_countries():
    """Descarga el shapefile del mundo una sola vez por proceso (antes se
    volvía a descargar en cada uno de los 3 subplots)."""
    try:
        import geopandas as gpd
    except ModuleNotFoundError:
        return None
    try:
        return gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/110m_cultural/"
            "ne_110m_admin_0_countries.zip"
        )
    except Exception:
        return None


def draw_geographic_context(ax, alpha: float = 0.7,
                           region_name: str = "uruguay",
                           dataset_root: Path | None = None) -> None:
    ax.set_xlim(-75.0, -34.0)
    ax.set_ylim(-56.0, -28.0)
    ax.set_aspect("equal")

    if dataset_root is not None:
        base = dataset_root / region_name
        if base.exists():
            try:
                lat2d, lon2d = compute_latlon_grid(str(base))
                region_mask = build_region_mask(lat2d, lon2d, region_name=region_name, base_path=str(base))
                if region_mask.size:
                    roi_cmap = ListedColormap(["#d9d9d9", "#6ecf8d"])
                    ax.pcolormesh(
                        lon2d, lat2d, region_mask.astype(np.uint8),
                        cmap=roi_cmap, shading="auto", alpha=0.18, vmin=0, vmax=1,
                    )
                    ax.contour(lon2d, lat2d, region_mask.astype(np.uint8), levels=[0.5], colors="#1b9e77", linewidths=1.0)
            except Exception:
                pass

    world = _load_world_countries()
    if world is None:
        lon0, lon1 = URUGUAY_LON_RANGE
        lat0, lat1 = URUGUAY_LAT_RANGE
        ax.add_patch(Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            facecolor="none", edgecolor="black", linewidth=1.2, alpha=alpha,
        ))
        ax.text(-56.8, -33.9, "Uruguay", fontsize=9, color="black", weight="bold")
        ax.text(-66.0, -31.5, "Argentina", fontsize=8, color="0.45")
        ax.text(-53.0, -29.0, "Brazil", fontsize=8, color="0.45")
        return

    admin_col = "ADMIN" if "ADMIN" in world.columns else "NAME"
    for admin, color, lw in (
        ("Uruguay", "black", 1.4),
        ("Argentina", "0.45", 0.9),
        ("Brazil", "0.45", 0.9),
    ):
        country = world[world[admin_col] == admin]
        if country.empty:
            continue
        country.boundary.plot(
            ax=ax,
            color=color,
            linewidth=lw,
            alpha=alpha,
        )
        country.plot(
            ax=ax,
            facecolor="none",
            edgecolor=color,
            linewidth=lw,
            alpha=alpha,
        )


def plot_audit(audit_dir: Path, output_path: Path | None = None,
              region_name: str = "uruguay", dataset_root: Path | None = None) -> Path:
    csv_path = audit_dir / "pixels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe {csv_path}")

    rows = read_pixels(csv_path)
    eco_points, fn_points, fp_points = build_plot_data(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    fig.suptitle(f"Audit: {audit_dir.name} | eco 150–153 / FN / FP", fontsize=12)

    eco_colors = {
        150: "#7f3b08",
        151: "#b35806",
        152: "#f1a340",
        153: "#fee0b6",
    }

    # Eco codes 150-153
    ax = axes[0]
    for code in (150, 151, 152, 153):
        pts = [(lat, lon) for lat, lon, c in eco_points if c == code]
        if not pts:
            continue
        lats = [p[0] for p in pts]
        lons = [p[1] for p in pts]
        ax.scatter(lons, lats, s=18, c=eco_colors[code], edgecolors="k", linewidths=0.2, label=f"eco {code}")
    draw_geographic_context(ax, region_name=region_name, dataset_root=dataset_root)
    ax.set_title("Códigos eco 150–153")
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.grid(True, alpha=0.25)
    if eco_points:
        ax.legend(loc="best", fontsize=8)

    # False negatives
    ax = axes[1]
    if fn_points:
        lats = [p[0] for p in fn_points]
        lons = [p[1] for p in fn_points]
        ax.scatter(lons, lats, s=28, c="#d62728", edgecolors="k", linewidths=0.3)
    draw_geographic_context(ax, region_name=region_name, dataset_root=dataset_root)
    ax.set_title(f"Falsos negativos ({len(fn_points)})")
    ax.set_xlabel("Lon")
    ax.grid(True, alpha=0.25)

    # False positives
    ax = axes[2]
    if fp_points:
        lats = [p[0] for p in fp_points]
        lons = [p[1] for p in fp_points]
        ax.scatter(lons, lats, s=28, c="#2ca02c", edgecolors="k", linewidths=0.3)
    draw_geographic_context(ax, region_name=region_name, dataset_root=dataset_root)
    ax.set_title(f"Falsos positivos ({len(fp_points)})")
    ax.set_xlabel("Lon")
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is None:
        output_path = audit_dir / "audit_pixels_overview.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Graficar códigos eco 150-153 y FN/FP para un audit.")
    parser.add_argument("audit", nargs="?", default=None, help="Nombre o índice de la auditoría (ej: 20260903_211636, 0, latest)")
    parser.add_argument("--audit", dest="audit_flag", default=None, help="Nombre o índice de la auditoría (alias)")
    parser.add_argument("--base-dir", type=Path, default=AUDIT_ROOT, help="Directorio base de auditorías (default: results/audit)")
    parser.add_argument("--dataset-root", type=Path, default=BASE_DIR / "dataset", help="Ruta base del dataset regional (default: dataset/)")
    parser.add_argument("--region", default="uruguay", help="Nombre de la región para dibujar el ROI (default: uruguay)")
    parser.add_argument("--out", type=Path, help="Ruta del PNG de salida (opcional)")
    args = parser.parse_args()

    audit_label = args.audit_flag if args.audit_flag is not None else args.audit
    if audit_label is None:
        audit_label = "latest"

    audit_dir = resolve_audit_dir(args.base_dir, audit_label)
    out_path = args.out if args.out else audit_dir / "audit_pixels_overview.png"

    csv_path = audit_dir / "pixels.csv"
    rows = read_pixels(csv_path)
    eco_points, fn_points, fp_points = build_plot_data(rows)

    image_path = plot_audit(audit_dir, out_path, region_name=args.region, dataset_root=args.dataset_root)
    print(f"Guardado: {image_path}")
    print(f"Eco 150-153: {len(eco_points)}")
    print(f"FN: {len(fn_points)}")
    print(f"FP: {len(fp_points)}")


if __name__ == "__main__":
    main()