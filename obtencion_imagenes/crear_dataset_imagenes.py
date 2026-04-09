from goes2go import GOES
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
import pyproj
import os

# ── Configuración ──────────────────────────────────────────────────────────────
SATELLITE   = 19
LAT_MIN, LAT_MAX = -35.5, -29.5
LON_MIN, LON_MAX = -59.0, -52.0

OUTPUT_DIR = "dataset_focos_igeos"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helper: recorte a Uruguay ────────────────────────────────
def goes_xy_slices(ds, lat_min, lat_max, lon_min, lon_max):
    proj_info  = ds["goes_imager_projection"]
    lon_origin = float(proj_info.longitude_of_projection_origin)
    H          = float(proj_info.perspective_point_height) + float(proj_info.semi_major_axis)

    crs_goes   = pyproj.CRS.from_dict({
        "proj": "geos", "lon_0": lon_origin,
        "h": float(proj_info.perspective_point_height),
        "a": float(proj_info.semi_major_axis),
        "b": float(proj_info.semi_minor_axis), "sweep": "x",
    })
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(4326), crs_goes, always_xy=True
    )
    px, py = transformer.transform(
        [lon_min, lon_max, lon_min, lon_max],
        [lat_min, lat_min, lat_max, lat_max]
    )
    x_vals, y_vals = ds["x"].values, ds["y"].values
    x_slice = slice(
        float(x_vals[np.argmin(np.abs(x_vals - np.nanmin(px) / H))]),
        float(x_vals[np.argmin(np.abs(x_vals - np.nanmax(px) / H))]),
    )
    y_slice = slice(
        float(y_vals[np.argmin(np.abs(y_vals - np.nanmax(py) / H))]),
        float(y_vals[np.argmin(np.abs(y_vals - np.nanmin(py) / H))]),
    )
    return x_slice, y_slice


# ── Objetos GOES para ambos productos ─────────────────────────────────────────
G_rad = GOES(satellite=SATELLITE, product="ABI-L1b-Rad",  domain="F", bands=7)
G_fdc = GOES(satellite=SATELLITE, product="ABI-L2-FDCF",  domain="F")


# ── Función principal: descarga un par (radiancia + máscara) para un timestamp ─
def descargar_muestra(timestamp: datetime):
    """
    Descarga la radiancia B7 y la máscara FDC para un timestamp dado.
    Devuelve un dict con arrays recortados a Uruguay, o None si falla.
    """
    try:
        ds_rad = G_rad.nearesttime(timestamp)
        ds_fdc = G_fdc.nearesttime(timestamp)

        # Recorte espacial
        xs, ys       = goes_xy_slices(ds_rad, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        rad_uy       = ds_rad.sel(x=xs, y=ys)["Rad"].values.astype(np.float32)

        xs2, ys2     = goes_xy_slices(ds_fdc, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        dqf_uy       = ds_fdc.sel(x=xs2, y=ys2)["DQF"].values.astype(np.int8)

        # El FDC tiene menor resolución (2km) que L1b B7 (2km también) → ok
        # Si hubiera diferencia de shape, reescalar con np o scipy

        ts_str = timestamp.strftime("%Y%m%d_%H%M")
        return {
            "timestamp": ts_str,
            "radiancia":  rad_uy,   # shape (H, W) float32
            "mascara_dqf": dqf_uy,  # shape (H, W) int8 — etiqueta
        }

    except Exception as e:
        print(f"   [WARN] Error en {timestamp}: {e}")
        return None


# ── Loop de descarga para un rango de fechas ──────────────────────────────────
def construir_dataset(fecha_inicio: str, fecha_fin: str, intervalo_horas: int = 1):
    """
    Descarga muestras entre fecha_inicio y fecha_fin cada `intervalo_horas`.
    Guarda cada muestra como un .npz en OUTPUT_DIR.
    """
    inicio = datetime.strptime(fecha_inicio, "%Y-%m-%d %H:%M")
    fin    = datetime.strptime(fecha_fin,    "%Y-%m-%d %H:%M")
    delta  = timedelta(hours=intervalo_horas)

    timestamps = []
    t = inicio
    while t <= fin:
        timestamps.append(t)
        t += delta

    print(f"Descargando {len(timestamps)} muestras...")
    registros = []

    for t in timestamps:
        print(f"  → {t.strftime('%Y-%m-%d %H:%M')} UTC", end=" ")
        muestra = descargar_muestra(t)

        if muestra is not None:
            nombre = os.path.join(OUTPUT_DIR, f"sample_{muestra['timestamp']}.npz")
            np.savez_compressed(
                nombre,
                radiancia   = muestra["radiancia"],
                mascara_dqf = muestra["mascara_dqf"],
            )
            n_fuego = np.sum(muestra["mascara_dqf"] == 0)
            registros.append({
                "timestamp": muestra["timestamp"],
                "archivo":   nombre,
                "pixeles_fuego": int(n_fuego),
                "shape": str(muestra["radiancia"].shape),
            })
            print(f"✓  focos detectados: {n_fuego}")
        else:
            print("✗")

    # Guardar índice del dataset
    df = pd.DataFrame(registros)
    df.to_csv(os.path.join(OUTPUT_DIR, "indice_dataset.csv"), index=False)
    print(f"\nDataset guardado en '{OUTPUT_DIR}/' — {len(df)} muestras válidas")
    print(df.groupby(df["pixeles_fuego"].gt(0))["timestamp"].count().rename({False:"sin fuego", True:"con fuego"}))
    return df


# ── Ejemplo de uso ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Período de ejemplo: una semana, cada hora (ajustá según tus necesidades)
    df = construir_dataset(
        fecha_inicio  = "2025-09-01 12:00",   # época seca en Uruguay → más incendios
        fecha_fin     = "2025-09-02 23:00",
        intervalo_horas = 1,
    )