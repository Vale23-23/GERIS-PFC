"""
downloader.py — config-driven download and spatial crop for GOES data.
"""

import os
import numpy as np
import pyproj
from goes2go import GOES

# Bandas de alta resolución que no pueden cargarse con nearesttime() completo
# porque el full disk pesa demasiado (21696x21696 píxeles)
HIGH_RES_BANDS = {2}


def goes_xy_slices(ds, lat_min, lat_max, lon_min, lon_max):
    """Convert a lat/lon bounding box to GOES x/y slices."""
    proj_info  = ds["goes_imager_projection"]
    lon_origin = float(proj_info.longitude_of_projection_origin)
    H          = float(proj_info.perspective_point_height) + float(proj_info.semi_major_axis)

    crs_goes = pyproj.CRS.from_dict({
        "proj": "geos", "lon_0": lon_origin,
        "h": float(proj_info.perspective_point_height),
        "a": float(proj_info.semi_major_axis),
        "b": float(proj_info.semi_minor_axis),
        "sweep": "x",
    })
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(4326), crs_goes, always_xy=True
    )
    px, py = transformer.transform(
        [lon_min, lon_max, lon_min, lon_max],
        [lat_min, lat_min, lat_max, lat_max],
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


def get_goes2go_cache_dir():
    """Devuelve el directorio donde goes2go guarda los archivos descargados."""
    try:
        from goes2go.config import config as g2g_config
        return g2g_config.get("default", {}).get("save_dir", os.path.expanduser("~/data"))
    except Exception:
        return os.path.expanduser("~/data")

def download_and_save(timestamp, product_cfg, region_cfg, satellite, domain, output_root):
    """
    Download a single product/timestamp, crop to region, save as .npy.
    For IR bands (7, 13, 14, 15), also saves Planck calibration coefficients
    as a parallel JSON file.

    Returns a dict with status and metadata, never raises.
    """
    import time
    import json

    PLANCK_BANDS = {7, 13, 14, 15} # Bandas IR donde nos importa la temperatura de brillo

    product_id = product_cfg["id"]
    folder     = os.path.join(output_root, product_id)
    os.makedirs(folder, exist_ok=True)

    file_path   = os.path.join(folder, f"{timestamp.strftime('%Y%m%d_%H%M')}.npy")
    coeffs_path = os.path.join(folder, f"{timestamp.strftime('%Y%m%d_%H%M')}_planck.json")

    band = product_cfg.get("band")

    # ¿Hace falta descargar la imagen?
    need_npy = not os.path.exists(file_path)

    # ¿Hace falta generar el JSON con los coeficientes Planck?
    need_json = (
        band in PLANCK_BANDS
        and not os.path.exists(coeffs_path)
    )

    # Si ya existe todo lo necesario, no hacemos nada.
    if not need_npy and not need_json:
        return {
            "status": "exists",
            "path": file_path,
            "product": product_id,
            "timestamp": timestamp.strftime("%Y%m%d_%H%M"),
        }

    print(f"    📥 Descargando {timestamp.strftime('%Y%m%d_%H%M')} {product_id}...", end="", flush=True)
    start_time = time.time()

    try:
        if band in HIGH_RES_BANDS:
            data = download_highres(timestamp, product_cfg, region_cfg, satellite, domain)
            planck_coeffs = None # Si el .npy existe pero falta el JSON de coeficientes (banda IR), lo señalamos
        else:
            g  = GOES(satellite=satellite, product=product_cfg["product"], domain=domain,
                    bands=band if band else None)
            ds = g.nearesttime(timestamp)
            xs, ys = goes_xy_slices(ds, **region_cfg)
            ds_cropped = ds.sel(x=xs, y=ys)
            data = ds_cropped[product_cfg["variable"]].values

            # Geometría: una sola vez por producto/banda (la grilla x/y es estática
            # en el tiempo; lo único que cambia por resolución es el band/product_id,
            # así que la cacheamos a nivel de carpeta de producto, no por timestamp)
            geom_path = os.path.join(folder, "geometry.json")
            if not os.path.exists(geom_path):
                proj_info = ds_cropped["goes_imager_projection"]
                geom_meta = {
                    "x": ds_cropped["x"].values.tolist(),
                    "y": ds_cropped["y"].values.tolist(),
                    "longitude_of_projection_origin": float(proj_info.longitude_of_projection_origin),
                    "perspective_point_height": float(proj_info.perspective_point_height),
                    "semi_major_axis": float(proj_info.semi_major_axis),
                    "semi_minor_axis": float(proj_info.semi_minor_axis),
                    "sweep_angle_axis": "x",
                }
                with open(geom_path, "w") as f:
                    json.dump(geom_meta, f)

            planck_coeffs = None
            if band in PLANCK_BANDS:
                planck_coeffs = {
                    "planck_fk1": float(ds["planck_fk1"].values),
                    "planck_fk2": float(ds["planck_fk2"].values),
                    "planck_bc1": float(ds["planck_bc1"].values),
                    "planck_bc2": float(ds["planck_bc2"].values),
                }

            ds.close()

        if data.size == 0:
            elapsed = time.time() - start_time
            print(f" ⚠️  vacío ({elapsed:.1f}s)")
            return {"status": "empty", "path": None, "product": product_id,
                    "timestamp": timestamp.strftime("%Y%m%d_%H%M")}

        # Guardar la imagen solo si hacía falta descargarla
        if need_npy:
            dtype = np.float32 if product_cfg["dtype"] == "float32" else np.int8
            np.save(file_path, data.astype(dtype))

        # Guardar el JSON solo si hacía falta
        if need_json and planck_coeffs is not None:
            with open(coeffs_path, "w") as f:
                json.dump(planck_coeffs, f)

        elapsed = time.time() - start_time
        print(f" ✅ ({elapsed:.1f}s, shape={list(data.shape)})")
        return {"status": "downloaded", "path": file_path, "product": product_id,
                "timestamp": timestamp.strftime("%Y%m%d_%H%M"), "shape": list(data.shape)}

    except Exception as e:
        elapsed = time.time() - start_time
        print(f" ❌ ({elapsed:.1f}s)")

        s3_path    = None
        status_msg = "error_local"
        error_detail = f"Error descargando ({elapsed:.1f}s): {str(e)}"

        try:
            exists_on_s3, s3_path = check_s3_exists(product_cfg, timestamp)
            if not exists_on_s3:
                status_msg   = "error_aws_gap"
                error_detail = f"Data Gap en AWS: no hay datos para {product_id} en s3://{s3_path}"
            else:
                status_msg   = "error_local"
                error_detail = f"El archivo existe en S3 pero falló la descarga ({elapsed:.1f}s): {str(e)}"
        except Exception as s3_err:
            error_detail += f" | S3 check también falló: {str(s3_err)}"

        return {
            "status":     "error",
            "substatus":  status_msg,
            "product":    product_id,
            "timestamp":  timestamp.strftime("%Y%m%d_%H%M"),
            "error":      error_detail,
            "product_path": s3_path,
        }