import os
import numpy as np
import pyproj  # ← ESTE FALTABA — causa de todos los errores
from datetime import datetime, timedelta
from goes2go import GOES
from concurrent.futures import ThreadPoolExecutor

# --- Configuración ---
SATELLITE = 19
LAT_MIN, LAT_MAX = -35.5, -29.5
LON_MIN, LON_MAX = -59.0, -52.0
OUTPUT_ROOT = "dataset_focos_igeos"


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

        xs, ys       = goes_xy_slices(ds_rad, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        rad_uy       = ds_rad.sel(x=xs, y=ys)["Rad"].values.astype(np.float32)

        xs2, ys2     = goes_xy_slices(ds_fdc, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        dqf_uy       = ds_fdc.sel(x=xs2, y=ys2)["DQF"].values.astype(np.int8)

        ts_str = timestamp.strftime("%Y%m%d_%H%M")
        return {
            "timestamp": ts_str,
            "radiancia":  rad_uy,
            "mascara_dqf": dqf_uy,
        }

    except Exception as e:
        print(f"   [WARN] Error en {timestamp}: {e}")
        return None


def descargar_y_guardar(timestamp, producto, banda=None):
    sub_dir = f"{producto}-B{banda:02d}" if banda else producto
    folder = os.path.join(OUTPUT_ROOT, sub_dir)
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{timestamp.strftime('%Y%m%d_%H%M')}.npy")

    if os.path.exists(file_path):
        return f"✅ YA EXISTE: {file_path}"

    try:
        g = GOES(satellite=SATELLITE, product=producto, domain="F", bands=banda)
        ds = g.nearesttime(timestamp)

        xs, ys = goes_xy_slices(ds, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

        var_name = "Rad" if "L1b" in producto else "DQF"
        data = ds.sel(x=xs, y=ys)[var_name].values

        if data.size == 0:
            return f"⚠️ VACÍO: {timestamp} no tiene datos en el área elegida."

        np.save(file_path, data.astype(np.float32 if var_name == "Rad" else np.int8))

        print(f"💾 GUARDADO EXITOSO: {file_path} | Shape: {data.shape}")
        return f"DESCARGADO: {file_path}"

    except Exception as e:
        print(f"❌ ERROR en {timestamp}: {e}")
        return f"ERROR: {e}"


def pipeline_paralelo(fecha_inicio, fecha_fin, productos_config, max_workers=4):
    """
    productos_config: lista de tuplas [(producto, banda), ...]
    """
    inicio = datetime.strptime(fecha_inicio, "%Y-%m-%d %H:%M")
    fin    = datetime.strptime(fecha_fin,    "%Y-%m-%d %H:%M")

    tareas = []
    t = inicio
    while t <= fin:
        for prod, banda in productos_config:
            tareas.append((t, prod, banda))
        t += timedelta(hours=1)

    print(f"Iniciando descarga paralela de {len(tareas)} archivos...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        resultados = list(executor.map(lambda p: descargar_y_guardar(*p), tareas))

    for r in resultados[-5:]:
        print(r)


# --- Ejemplo de uso ---
if __name__ == "__main__":
    config_hoy = [("ABI-L1b-Rad", 7), ("ABI-L2-FDCF", None)]
    pipeline_paralelo("2025-09-01 11:00", "2025-09-02 12:00", config_hoy)