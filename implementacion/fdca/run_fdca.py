"""
Script to run FDCA on a timestamp downloaded by pipeline.py
and generate figures that show effect of part I and part II.

Usage:
  python run_fdca.py --timestamp 20250905_1500 --region uruguay
  python run_fdca.py --timestamp 20250905_1500 --region uruguay --save-outputs

Outputs generated in figures/<timestamp>/:
  00_inputs.png          -> BT7, BT14 and SZA (algorithm inputs)
  01_part1_filters.png   -> Part I filter effects
  02_part2_confirm.png   -> Part II effects
  03_fire_map.png        -> Final map

Outputs generated in data/<timestamp>/ (with --save-outputs):
  fire_mask.npy
  fail_char.npy
  summary.json
"""

import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from fdca.dataset import default_dataset_root
# ── Arguments ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run FDCA on GOES-19 data")

parser.add_argument("--timestamp", required=True,
    help='Timestamp to process, e.g. "20250905_1500"')
parser.add_argument("--region", default="uruguay",
    help="Region from config.yaml (default: uruguay)")
parser.add_argument("--dataset-root", default=default_dataset_root(),
    help="Dataset root folder (default: $GERIS_DATASET_ROOT env var if set, else './dataset')")
parser.add_argument("--config",
    default=str(Path(__file__).resolve().parent / "config.yaml"),
    help="Path to config.yaml (default: config.yaml bundled inside fdca/)")
parser.add_argument("--download", action="store_true",
    help="Download the requested scene from Hugging Face when it is missing")
parser.add_argument("--save-outputs", action="store_true",
    help="Save fire_mask.npy, fail_char.npy, and summary.json")
parser.add_argument("--output-dir", default="figures",
    help="Base folder for figures (default: figures)")
parser.add_argument("--state-path", default="data/prev_fire_mask.npy",
    help="Path for temporal fire history (default: data/prev_fire_mask.npy)")
parser.add_argument("--reference-mask", default=None,
    help="Path to the final reference mask .npy; inferred from dataset-root when omitted")

args = parser.parse_args()

TS      = args.timestamp
REGION  = args.region
FIG_DIR = Path(args.output_dir) / TS
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Import FDCA constants to build the palette ────────────────
sys.path.insert(0, str(Path(args.config).parent.parent))
from fdca.constants import FireMask
from fdca.metrics import evaluate, print_metrics

TEMPORAL_FIRE_CODES = (
    FireMask.TEMP_PROCESSED, FireMask.TEMP_SATURATED,
    FireMask.TEMP_CLOUD, FireMask.TEMP_HIGH,
    FireMask.TEMP_MED, FireMask.TEMP_LOW,
)

# ──  fire_mask codes (Table 3.11 ATBD) ───────────────────────────────────────
FM_PALETTE = {
    # Parte I — no procesado / bloqueo geométrico
    FireMask.NON_PROCESSED:        "#f5f5f5",  # 0   no debería quedar en output final
    FireMask.SPACE:                "#dde8f0",  # 40  espacio
    FireMask.ZENITH_BLOCK:         "#c8dce8",  # 50  SZA > 80°
    FireMask.GLINT_BLOCK:          "#b8cfe0",  # 60  glint

    # Parte I — fire-free / demasiado frío
    FireMask.FIRE_FREE:            "#e8f0e0",  # 100 pasó todos los tests, no es fuego
    FireMask.TOO_COLD:             "#d8e8d0",  # 201 demasiado frío

    # Parte I — datos malos / faltantes
    FireMask.MISS_CH7:             "#f0d0a8",  # 120
    FireMask.MISS_CH14:            "#f0c898",  # 121
    FireMask.SAT_CH7:              "#e8d080",  # 123 saturado ch7 (+buffer)
    FireMask.SAT_CH14:             "#e8c060",  # 124 saturado ch14 (+buffer)
    FireMask.NEG_RAD:              "#d8b088",  # 125 radiancia negativa
    FireMask.UNUS_CH7:             "#c8a078",  # 126 ch7 < 200K
    FireMask.UNUS_CH14:            "#b89068",  # 127 ch14 < 200K

    # Parte I — ecosistema / superficie inválida
    FireMask.BAD_ECOSYSTEM:        "#a8bfd8",  # 150 agua/desierto/vecino inválido
    FireMask.SEA_WATER:            "#98afc8",  # 151 USGS sea water
    FireMask.COAST_FRINGE:         "#88a0c0",  # 152 USGS coastline fringe
    FireMask.INLAND_WATER:         "#b8d0e8",  # 153 USGS inland water

    # Parte I — background / corrección fallida
    FireMask.NO_BACKGROUND:        "#c0c0c0",  # 170 no se determinó background
    FireMask.CONV_ERROR:           "#a8a8a8",  # 180 BT/radiancia corregida ≤ 0

    # Parte I — nubes
    FireMask.CLOUD_BT14:           "#909090",  # 200
    FireMask.CLOUD_BT7_BT14_NEG:   "#989898",  # 205
    FireMask.CLOUD_BT7_BT14_POS:   "#a0a0a0",  # 210
    FireMask.CLOUD_ALBEDO:         "#a8a8a8",  # 215
    FireMask.CLOUD_BT15:           "#b0b0b0",  # 220
    FireMask.CLOUD_BT14_BT15_NEG:  "#b8b8b8",  # 225
    FireMask.CLOUD_BT14_BT15_POS:  "#c0c0c0",  # 230

    # Parte I — reflectividad along-scan (cerca de borde de nube)
    FireMask.ALONG_SCAN_NIGHT:     "#d0c0e0",  # 240
    FireMask.ALONG_SCAN_DAY:       "#d8c8e8",  # 245

    # Parte II — fuego confirmado (sin historial temporal)
    FireMask.PROCESSED:            "#ff4500",  # 10
    FireMask.SATURATED:            "#ff6347",  # 11
    FireMask.CLOUD_CONTAM:         "#909090",  # 12
    FireMask.HIGH_PROB:            "#e60000",  # 13
    FireMask.MED_PROB:             "#ff8c00",  # 14
    FireMask.LOW_PROB:             "#ffd700",  # 15

    # Parte II — fuego confirmado con historial temporal (+20)
    FireMask.TEMP_PROCESSED:       "#990000",  # 30
    FireMask.TEMP_SATURATED:       "#aa3300",  # 31
    FireMask.TEMP_CLOUD:           "#505050",  # 32
    FireMask.TEMP_HIGH:            "#880000",  # 33
    FireMask.TEMP_MED:             "#aa5500",  # 34
    FireMask.TEMP_LOW:             "#aa8800",  # 35
}

def mask_to_rgb(fire_mask):
    H, W = fire_mask.shape
    img = np.ones((H, W, 3))
    for code, color in FM_PALETTE.items():
        rgb = mcolors.to_rgb(color)
        img[fire_mask == code] = rgb
    return img


# ── Figure 0: reference inputs ───────────────────────────────────────────

def fig_inputs(inp, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Inputs GOES-19 | {REGION} | {TS}", fontsize=12, fontweight="bold")

    # BT7
    v7 = np.nanpercentile(inp.bt7, [2, 98])
    im0 = axes[0].imshow(inp.bt7, cmap="inferno", vmin=v7[0], vmax=v7[1], origin="upper")
    axes[0].set_title("BT Canal 7 (3.9 µm) [K]\n← sensible a fuego (alta T → brillo)")
    plt.colorbar(im0, ax=axes[0], label="K", shrink=0.8)

    # BT14
    v14 = np.nanpercentile(inp.bt14, [2, 98])
    im1 = axes[1].imshow(inp.bt14, cmap="RdYlBu_r", vmin=v14[0], vmax=v14[1], origin="upper")
    axes[1].set_title("BT Canal 14 (11.2 µm) [K]\n← referencia térmica de fondo")
    plt.colorbar(im1, ax=axes[1], label="K", shrink=0.8)

    # SZA 
    im2 = axes[2].imshow(inp.sza, cmap="twilight", vmin=0, vmax=120, origin="upper")
    axes[2].set_title("Ángulo zenith solar [°]\n< 85° = día  |  ≥ 85° = noche")
    plt.colorbar(im2, ax=axes[2], label="°", shrink=0.8)
    # Contorno de la frontera día/noche
    try:
        axes[2].contour(inp.sza, levels=[85], colors="white", linewidths=1.5)
    except Exception:
        pass

    for ax in axes:
        ax.set_xlabel("Columna")
        ax.set_ylabel("Fila")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path.name}")


# ── Figure 1:  Part I effects ───────────────────────────────────────────

def fig_part1(inp, fire_mask_p1, fail_char_p1, candidates, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Parte I — Filtros píxel a píxel (ATBD 3.4.2.1–3.4.2.13)\n"
                 f"{REGION} | {TS}", fontsize=11, fontweight="bold")

        # Panel A: estado al final de Parte I
    # fire_mask_p1 NUNCA contiene 10-15/30-35 (esos códigos los asigna
    # Parte II en 3.4.2.15). Los candidatos de Parte I se identifican por
    # coordenada (i, j) en la lista `candidates`, no por el valor del mask.
    cat = np.ones_like(fire_mask_p1)  # default: descartado

    blocked_codes = [FireMask.SPACE, FireMask.ZENITH_BLOCK, FireMask.GLINT_BLOCK]
    cat[np.isin(fire_mask_p1, blocked_codes)] = 0   # sin datos / bloqueado

    for c in candidates:
        cat[c.i, c.j] = 2  # candidato de Parte I (aún sin clasificar)

    colors_cat = ["#dde8f0", "#b0c4de", "#ff4500"]
    cmap_cat   = mcolors.ListedColormap(colors_cat)
    axes[0].imshow(cat, cmap=cmap_cat, vmin=0, vmax=2, origin="upper")
    axes[0].set_title(f"Estado al final de Parte I\n"
                      f"{(cat==2).sum()} candidatos de {cat.size:,} píxeles totales")
    patches_cat = [
        mpatches.Patch(color=colors_cat[0], label="Sin datos / bloqueado (espacio, SZA, glint)"),
        mpatches.Patch(color=colors_cat[1], label="Descartado (fire-free, nube, agua, ecosistema...)"),
        mpatches.Patch(color=colors_cat[2], label="Candidato fuego (Parte I)"),
    ]
    axes[0].legend(handles=patches_cat, loc="lower left", fontsize=7, framealpha=0.85)

    # Panel B: distribución de códigos de descarte (fail_char)
    FC_LABELS = {
        0: "Válido (ninguno)",   1: "F1: BT7−BT14 bajo",
        2: "F2: BT7−Fondo bajo", 3: "F3: BT post-corr",
        4: "F4: BT14−Fondo",     5: "F5: BT7−Fondo",
        6: "F6: Dozier falla",   7: "F7: Saturado",
        8: "F8: Glint/nube",     9: "F9: Glint Dozier",
       10: "F10: Nube+BT alto",
    }
    fc_vals = [c.fail_char for c in candidates]
    fc_counts = {}
    for v in fc_vals:
        fc_counts[v] = fc_counts.get(v, 0) + 1

    if fc_counts:
        keys = sorted(fc_counts)
        labels = [FC_LABELS.get(k, f"FC={k}") for k in keys]
        values = [fc_counts[k] for k in keys]
        colors_bar = plt.cm.tab10(np.linspace(0, 1, len(keys)))
        bars = axes[1].barh(labels, values, color=colors_bar)
        axes[1].set_xlabel("Candidatos")
        axes[1].set_title(f"FailChar de candidatos ({len(candidates)} total)\n"
                          f"(código que describe por qué pasó o qué corrección aplicó)")
        axes[1].invert_yaxis()
        for bar, v in zip(bars, values):
            axes[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                         str(v), va="center", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "Sin candidatos", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=13)

    # Panel C: BT7 con candidatos encima
    vbt = np.nanpercentile(inp.bt7, [2, 99])
    axes[2].imshow(inp.bt7, cmap="gray", vmin=vbt[0], vmax=vbt[1], origin="upper", alpha=0.8)
    if candidates:
        ci = [c.i for c in candidates]
        cj = [c.j for c in candidates]
        axes[2].scatter(cj, ci, c="red", s=10, marker="x",
                        linewidths=0.8, label=f"{len(candidates)} candidatos P1")
        axes[2].legend(loc="lower left", fontsize=8, framealpha=0.85)
    axes[2].set_title("BT Canal 7 + candidatos Parte I\n(cada × = píxel que pasó todos los filtros)")

    for ax in axes:
        ax.set_xlabel("Columna")
        ax.set_ylabel("Fila")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path.name}")


# ── Figura 2: efecto de la Parte II ──────────────────────────────────────────

def fig_part2(inp, fire_mask_p1, fire_mask_p2, candidates, confirmed, save_path):
    n_cand = len(candidates)
    n_conf = len(confirmed)
    n_elim = n_cand - n_conf

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Parte II — Confirmación, clasificación y filtro temporal (ATBD 3.4.2.14–3.4.2.18)\n"
                 f"{REGION} | {TS}", fontsize=11, fontweight="bold")

    # [0,0]: Fire mask final con paleta ATBD
    axes[0,0].imshow(mask_to_rgb(fire_mask_p2), origin="upper")
    axes[0,0].set_title("Fire Mask Final (Parte II)\ncódigos 10–15 = fuego / 30–35 = fuego+historial")
    fire_leg = [
        mpatches.Patch(color="#e60000", label="Alta prob (13/33)"),
        mpatches.Patch(color="#ff8c00", label="Media prob (14/34)"),
        mpatches.Patch(color="#ffd700", label="Baja prob (15/35)"),
        mpatches.Patch(color="#ff4500", label="Procesado (10/30)"),
        mpatches.Patch(color="#909090", label="Nublado (12/32)"),
        mpatches.Patch(color="#b0c4de", label="No fuego"),
    ]
    axes[0,0].legend(handles=fire_leg, loc="lower left", fontsize=7, framealpha=0.9)

    # [0,1]: Torta P1 candidatos vs P2 confirmados
    if n_cand > 0:
        axes[0,1].pie(
            [n_conf, n_elim],
            labels=[f"Confirmados P2\n({n_conf})", f"Eliminados P2\n({n_elim})"],
            colors=["#ff4500", "#b0c4de"],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )
        axes[0,1].set_title(f"Candidatos P1 → confirmados P2\n"
                             f"Parte II aplica tests adicionales para eliminar falsas alarmas")
    else:
        axes[0,1].text(0.5, 0.5, "Sin candidatos en P1", ha="center", va="center",
                       transform=axes[0,1].transAxes, fontsize=12)

    # [0,2]: Barras de categorías
    CATS = {
        "Procesado\n(10/30)": (10, 30),
        "Saturado\n(11/31)":  (11, 31),
        "Nublado\n(12/32)":   (12, 32),
        "Alta\n(13/33)":      (13, 33),
        "Media\n(14/34)":     (14, 34),
        "Baja\n(15/35)":      (15, 35),
    }
    cat_vals  = [int((fire_mask_p2==a).sum() + (fire_mask_p2==b).sum()) for a,b in CATS.values()]
    cat_cols  = ["#ff4500","#ff6347","#909090","#e60000","#ff8c00","#ffd700"]
    bars = axes[0,2].bar(list(CATS.keys()), cat_vals, color=cat_cols)
    axes[0,2].set_ylabel("Píxeles")
    axes[0,2].set_title("Distribución por categoría final\n(cada categoría tiene un umbral de confianza distinto)")
    axes[0,2].tick_params(axis="x", labelsize=7)
    for bar, v in zip(bars, cat_vals):
        if v > 0:
            axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                           str(v), ha="center", va="bottom", fontsize=9)

    # [1,0]: candidatos de Parte I que NO fueron confirmados en Parte II
    diff = np.zeros_like(fire_mask_p1, dtype=int)
    for c in candidates:
        diff[c.i, c.j] = -1  # candidato original
    for f in confirmed:
        diff[f.i, f.j] = 1   # confirmado
    im_diff = axes[1,0].imshow(diff, cmap="RdYlGn", vmin=-1, vmax=1, origin="upper")
    axes[1,0].set_title("Candidatos Parte I → confirmados Parte II\nverde = confirmado, rojo = eliminado")
    plt.colorbar(im_diff, ax=axes[1,0], shrink=0.8, ticks=[-1, 0, 1])
    n_elim_map = len(candidates) - len(confirmed)
    axes[1,0].set_xlabel(f"{n_elim_map} candidatos eliminados por Parte II (falsas alarmas)")

    # [1,1]: FRP de confirmados
    frps = [f.frp for f in confirmed if hasattr(f, "frp") and f.frp is not None and f.frp > 0]
    if frps:
        axes[1,1].hist(frps, bins=min(20, len(frps)), color="#ff4500",
                       edgecolor="white", linewidth=0.5)
        axes[1,1].axvline(np.median(frps), color="black", linestyle="--",
                           label=f"Mediana {np.median(frps):.1f} MW")
        axes[1,1].set_xlabel("FRP [MW]")
        axes[1,1].set_ylabel("Píxeles")
        axes[1,1].set_title(f"Fire Radiative Power (FRP)\n"
                             f"proxy de intensidad del fuego — máx: {max(frps):.1f} MW")
        axes[1,1].legend(fontsize=9)
    else:
        axes[1,1].text(0.5, 0.5, "Sin valores FRP\n(sin fuegos procesados con solución Dozier)",
                       ha="center", va="center", transform=axes[1,1].transAxes, fontsize=10)
        axes[1,1].set_title("Fire Radiative Power (FRP)")

    # [1,2]: Mapa final con puntos coloreados por categoría
    vbt = np.nanpercentile(inp.bt7, [2, 99])
    axes[1,2].imshow(inp.bt7, cmap="gray", vmin=vbt[0], vmax=vbt[1], origin="upper", alpha=0.7)
    CAT_COLORS = {10:"#ff4500",11:"#ff6347",12:"#909090",13:"#e60000",
                  14:"#ff8c00",15:"#ffd700",30:"#880000",31:"#aa3300",
                  32:"#505050",33:"#770000",34:"#aa5500",35:"#aa8800"}
    for code, color in CAT_COLORS.items():
        pts = [(f.j, f.i) for f in confirmed if fire_mask_p2[f.i, f.j] == code]
        if pts:
            xs, ys = zip(*pts)
            axes[1,2].scatter(xs, ys, c=color, s=20, marker="o",
                              linewidths=0, zorder=5, label=f"Código {code}")
    if confirmed:
        axes[1,2].legend(loc="lower left", fontsize=7, framealpha=0.85, ncol=2)
    axes[1,2].set_title(f"Fuegos confirmados sobre BT7\n({len(confirmed)} píxeles confirmados en Parte II)")

    # Marcar también cuántos tienen historial temporal (+20)
    n_temporal = int(np.isin(fire_mask_p2, TEMPORAL_FIRE_CODES).sum())
    if n_temporal > 0:
        axes[1,2].set_xlabel(f"{n_temporal} con historial temporal (código +20)")

    for ax in axes.flat:
        ax.set_xlabel(ax.get_xlabel() or "Columna")
        ax.set_ylabel(ax.get_ylabel() or "Fila")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path.name}")


# ── Figura 3: mapa georeferenciado ────────────────────────────────────────────

def fig_fire_map(inp, fire_mask_p2, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    lat, lon = inp.latitudes, inp.longitudes
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    ax.imshow(mask_to_rgb(fire_mask_p2),
              extent=extent, origin="upper", aspect="auto")
    ax.set_xlabel("Longitud [°]")
    ax.set_ylabel("Latitud [°]")
    ax.set_title(f"Fire Mask Final — GOES-19 | {REGION.upper()} | {TS}", fontsize=12)

    patches = [
        mpatches.Patch(color="#e60000", label="Alta probabilidad (13/33)"),
        mpatches.Patch(color="#ff8c00", label="Media probabilidad (14/34)"),
        mpatches.Patch(color="#ffd700", label="Baja probabilidad (15/35)"),
        mpatches.Patch(color="#ff4500", label="Procesado (10/30)"),
        mpatches.Patch(color="#b0c4de", label="Libre de fuego"),
        mpatches.Patch(color="#909090", label="Nublado/sin datos"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path.name}")



def build_temporal_state(
    timestamp: str,
    region: str,
    data_root: str,
    shape: tuple,
    download: bool = False,
) -> np.ndarray:
    """
    Build temporal state mask from previous 12 hours of fire detections.
    
    This function downloads previous scenes if needed and builds a temporal
    history mask for Part II.
    """
    from fdca.temporal_filter import TemporalFilter
    from fdca.dataset import ensure_timestamp_data
    
    # Define download callback that downloads missing scenes
    def download_callback(ts: str):
        if download:
            ensure_timestamp_data(
                timestamp=ts,
                region=region,
                dataset_root=data_root,
                download=True,
            )
    
    tf = TemporalFilter(
        data_root=data_root,
        region=region,
        timestamp=timestamp,
        shape=shape,
        lookback_hours=12,
        download_callback=download_callback if download else None,
    )
    
    return tf.load_previous_fires()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from fdca.part1 import run_part1
    from fdca.part2 import run_part2
    from fdca.algorithm import _to_epoch
    from fdca.state import PreviousFireMaskStore

    sys.path.insert(0, str(Path(__file__).parent)) # Import input adaptader
    from fdca.dataset import ensure_timestamp_data
    from fdca.fdca_adapter import load_fdca_input

    ensure_timestamp_data(
        timestamp=TS,
        region=REGION,
        dataset_root=args.dataset_root,
        download=args.download,
    )

    print("=" * 60)
    print(f"FDCA  |  GOES-19  |  {REGION}  |  {TS}")
    print("=" * 60)

    inp = load_fdca_input(    # Load inputs
        timestamp=TS, region=REGION,
        dataset_root=args.dataset_root,
        config_path=args.config,
    )

    print(f"\n[ Temporal ] Building temporal state from previous 12 hours...")
    
    # Build temporal state from previous detections
    temporal_state = build_temporal_state(
        timestamp=TS,
        region=REGION,
        data_root=args.dataset_root,
        shape=inp.bt7.shape,
        download=args.download,
    )
    
    # Use the temporal state in Part II
    inp.prev_fire_mask = temporal_state
    
    # Also save it to the state path for persistence
    state_store = PreviousFireMaskStore(inp.bt7.shape, args.state_path)
    state_store.data = temporal_state

    print(f"\n[ Figure 0 ] Reference inputs...")
    fig_inputs(inp, FIG_DIR / "00_inputs.png")

    # ── Part I ───────────────────────────────────────────────────────────────
    print(f"\n[ Part I ] Running pixel to pixel filters ...")
    t0 = datetime.now()
    fire_mask_p1, fail_char_p1, candidates = run_part1(
        bt7=inp.bt7, rad7=inp.rad7,
        bt14=inp.bt14, rad14=inp.rad14,
        bt13=inp.bt13, rad13=inp.rad13,
        bt15=inp.bt15, refl2=inp.refl2,
        latitudes=inp.latitudes, longitudes=inp.longitudes,
        sza=inp.sza, glint_angle=inp.glint_angle,
        lza=inp.lza, azimuth=inp.azimuth,
        tpw=inp.tpw, emiss7=inp.emiss7, emiss14=inp.emiss14,
        lut_tpw=inp.lut_tpw, FPT=inp.FPT,
        coeffs7=inp.coeffs7, coeffs14=inp.coeffs14, coeffs13=inp.coeffs13,
        land_mask=inp.land_mask,
        eco_mask=inp.eco_mask,
        data_quality=inp.data_quality,
    )
    t1 = datetime.now()
    print(f"  -> {len(candidates)} candidates  ({(t1-t0).total_seconds():.1f}s)")

    print(f"\n[ Figure 1 ] Part I effect ...")
    fig_part1(inp, fire_mask_p1, fail_char_p1, candidates, FIG_DIR / "01_part1_filters.png")

    # ── Part II ──────────────────────────────────────────────────────────────
    print(f"\n[ Part II ] Confirmation and clasification ...")
    fire_mask_p2, fail_char_p2, confirmed = run_part2(
        candidates=candidates,
        fire_mask=fire_mask_p1.copy(),
        fail_char_arr=fail_char_p1.copy(),
        prev_fire_mask=inp.prev_fire_mask,
        current_epoch=_to_epoch(inp.scan_time),
    )
    t2 = datetime.now()
    print(f"  -> {len(confirmed)} confirmed  ({(t2-t1).total_seconds():.1f}s)")

    # Persist only after Part II has completed, so the next scene can apply
    # the temporal filter to the final detections of this scene.
    state_store.update(fire_mask_p2, _to_epoch(inp.scan_time))
    state_store.save()

    # ── Metrics against the final reference mask ─────────────────────────────
    reference_path = (Path(args.reference_mask) if args.reference_mask else
                      Path(args.dataset_root) / REGION / "ABI-L2-FDCF-Mask" / f"{TS}.npy")
    metrics = None
    if reference_path.exists():
        reference_mask = np.load(reference_path).astype(np.uint8)
        candidate_mask = np.zeros_like(fire_mask_p2, dtype=bool)
        for candidate in candidates:
            candidate_mask[candidate.i, candidate.j] = True
        metrics = evaluate(reference_mask, fire_mask_p1, candidate_mask, fire_mask_p2)
        print_metrics(metrics, str(reference_path))
    else:
        print(f"\nNo se encontró máscara de referencia; métricas omitidas: {reference_path}")

    print(f"\n[ Figure 2 ] Part II effect ...")
    fig_part2(inp, fire_mask_p1, fire_mask_p2, candidates, confirmed,
              FIG_DIR / "02_part2_confirm.png")

    print(f"\n[ Figure 3 ] Map ...")
    fig_fire_map(inp, fire_mask_p2, FIG_DIR / "03_fire_map.png")

    if args.save_outputs: #  Save outputs
        out_dir = Path("data") / TS
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "fire_mask.npy",  fire_mask_p2)
        np.save(out_dir / "fail_char.npy",  fail_char_p2)

        frps = [f.frp for f in confirmed if f.frp is not None and f.frp > 0]
        summary = {
            "timestamp":    TS,
            "region":       REGION,
            "shape":        list(inp.bt7.shape),
            "n_candidates": len(candidates),
            "n_confirmed":  len(confirmed),
            "n_high":    int(((fire_mask_p2==13)|(fire_mask_p2==33)).sum()),
            "n_medium":  int(((fire_mask_p2==14)|(fire_mask_p2==34)).sum()),
            "n_low":     int(((fire_mask_p2==15)|(fire_mask_p2==35)).sum()),
            "n_temporal":int(np.isin(fire_mask_p2, TEMPORAL_FIRE_CODES).sum()),
            "frp_median_mw": float(np.median(frps)) if frps else None,
            "frp_max_mw":    float(np.max(frps))    if frps else None,
            "run_time_s":    (t2-t0).total_seconds(),
        }
        if metrics is not None:
            summary["metrics"] = metrics
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Outputs guardados en {out_dir}/")

    # ── Summary ────────────────────────────────────────────────────────────────
    frps = [f.frp for f in confirmed if f.frp is not None and f.frp > 0]
    print(f"\n{'='*60}")
    print(f"  Candidates Part I : {len(candidates)}")
    print(f"  Confirmed Part II: {len(confirmed)}")
    print(f"  High prob (13/33)  : {((fire_mask_p2==13)|(fire_mask_p2==33)).sum()}")
    print(f"  Medium prob (14/34) : {((fire_mask_p2==14)|(fire_mask_p2==34)).sum()}")
    print(f"  Low prob  (15/35) : {((fire_mask_p2==15)|(fire_mask_p2==35)).sum()}")
    print(f"  With history (+20): {np.isin(fire_mask_p2, TEMPORAL_FIRE_CODES).sum()}")
    if frps:
        print(f"  FRP median        : {np.median(frps):.1f} MW")
        print(f"  FRP maximum         : {np.max(frps):.1f} MW")
    print(f"  Total time       : {(t2-t0).total_seconds():.1f}s")
    print(f"  Saved figures  : {FIG_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
