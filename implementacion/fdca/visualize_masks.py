"""
visualize_masks.py
───────────────────
Visualiza las máscaras que determinan qué píxeles se procesan en el
pipeline FDCA: region_mask (recorte geográfico de la región configurada,
p.ej. Uruguay), eco_mask (tierra/agua/costa) y land_mask (combinación
final que usa part1.py).

Uso:
    python visualize_masks.py --base dataset/uruguay --region uruguay

    # o importado desde un notebook / script:
    from visualize_masks import plot_masks
    plot_masks(base_path="dataset/uruguay", region_name="uruguay")
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    from fdca.fdca_adapter import (
        compute_latlon_grid,
        load_geometry,
        build_region_mask,
        build_surface_masks,
    )
except ModuleNotFoundError:  # ejecución directa desde la carpeta fdca/
    from fdca_adapter import (
        compute_latlon_grid,
        load_geometry,
        build_region_mask,
        build_surface_masks,
    )


def plot_masks(base_path: str, region_name: str = "uruguay", savepath: str | None = None):
    # 1. Grilla real de lat/lon (misma fuente que usa el adapter)
    lat2d, lon2d = compute_latlon_grid(base_path)

    # 2. Máscara de región (el recorte geográfico, p.ej. Uruguay)
    #    OJO: hay que pasar base_path, si no nunca busca el geojson local
    #    (country_mask.geojson / ne_110m_admin_0_countries.geojson, etc.)
    #    y siempre termina en el bbox rectangular o intentando descargar.
    region_mask = build_region_mask(lat2d, lon2d, region_name=region_name, base_path=base_path)

    # 3. Máscaras de superficie completas (incluye land_mask final)
    masks = build_surface_masks(lat2d, lon2d, region_name=region_name, base_path=base_path)
    land_mask = masks["land_mask"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)

    panels = [
        (region_mask, f"region_mask\n({region_name})", "Greens"),
        (masks["land_cover"] != 0, "land_cover != 0\n(tierra, sin filtrar por región)", "Oranges"),
        (land_mask, "land_mask final\n(region_mask & eco==0)", "Blues"),
    ]

    for ax, (mask, title, cmap) in zip(axes, panels):
        # pcolormesh ubica cada píxel en su lat/lon real (lon2d, lat2d),
        # sin asumir que la grilla es regular ni tener que adivinar
        # orientación de filas/columnas como haría imshow+extent+flipud.
        im = ax.pcolormesh(
            lon2d, lat2d, mask.astype(np.uint8),
            cmap=cmap, vmin=0, vmax=1, shading="auto",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitud [°]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2, linestyle="--")

    axes[0].set_ylabel("Latitud [°]")
    fig.suptitle(f"Máscaras del pipeline FDCA — región '{region_name}'", fontsize=12)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Guardado en: {savepath}")
    else:
        plt.show()

    # Resumen numérico rápido
    total = region_mask.size
    print(f"\nShape grilla: {region_mask.shape}")
    print(f"region_mask : {region_mask.sum():>8} / {total} píxeles dentro ({100*region_mask.mean():.1f}%)")
    print(f"land_mask   : {land_mask.sum():>8} / {total} píxeles a procesar ({100*land_mask.mean():.1f}%)")

    return region_mask, land_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualiza las máscaras de recorte del pipeline FDCA")
    parser.add_argument("--base", required=True, help="Ruta base de la región, ej. dataset/uruguay")
    parser.add_argument("--region", default="uruguay", help="Nombre de la región (default: uruguay)")
    parser.add_argument("--save", default=None, help="Ruta para guardar el PNG en vez de mostrarlo")
    args = parser.parse_args()

    plot_masks(args.base, args.region, args.save)