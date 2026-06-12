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


def download_highres(timestamp, product_cfg, region_cfg, satellite, domain):
    """
    Descarga y cropea bandas de alta resolución (ej: B02) usando xr.open_dataset
    en lugar de nearesttime(), para evitar cargar el full disk en memoria.
    """
    import xarray as xr

    band = product_cfg.get("band")
    g = GOES(satellite=satellite, product=product_cfg["product"], domain=domain,
             bands=band) #GOES NO TIENE TIMEOUT COMO PARAMETRO!!!

    # Solo obtener la lista de archivos, sin abrirlos
    files_df = g.nearesttime(timestamp, return_as="filelist")
    if files_df is None or len(files_df) == 0:
        raise RuntimeError(f"goes2go no encontró archivos para {product_cfg['id']} en {timestamp}")

    # Construir el path completo al archivo local
    relative_path = files_df.iloc[0]["file"]
    cache_dir = get_goes2go_cache_dir()
    full_path = os.path.join(cache_dir, relative_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Archivo descargado no encontrado en: {full_path}")

    # Abrir lazy (sin cargar en RAM) y cropear solo la región
    ds = xr.open_dataset(full_path, engine="h5netcdf")
    xs, ys = goes_xy_slices(ds, **region_cfg)
    data = ds.sel(x=xs, y=ys)[product_cfg["variable"]].values
    ds.close()

    return data


def check_s3_exists(product_cfg, timestamp):
    """Verifica existencia en S3 usando boto3 con acceso anónimo al bucket público."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    julian_day = timestamp.strftime('%j')
    hour       = timestamp.strftime('%H')
    year       = timestamp.strftime('%Y')

    s3_product = product_cfg["product"]
    if s3_product == "ABI-L1b-Rad":
        s3_product = "ABI-L1b-RadF"

    prefix = f"{s3_product}/{year}/{julian_day}/{hour}/"
    bucket = "noaa-goes19"

    try:
        s3 = boto3.client(
            's3',
            config=Config(
                signature_version=UNSIGNED,  # bucket público, sin autenticación
                connect_timeout=5,
                read_timeout=5,
                retries={'max_attempts': 1},
            ),
            region_name='us-east-1'
        )
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=5,
        )
        files = response.get('Contents', [])
        if product_cfg.get("band"):
            band_str = f"M6C{product_cfg['band']:02d}"
            exists = any(band_str in f['Key'] for f in files)
        else:
            exists = len(files) > 0

        return exists, f"{bucket}/{prefix}"
    except Exception:
        return False, f"{bucket}/{prefix}"


def download_and_save(timestamp, product_cfg, region_cfg, satellite, domain, output_root):
    """
    Download a single product/timestamp, crop to region, save as .npy.

    Returns a dict with status and metadata, never raises.
    """
    import time

    product_id = product_cfg["id"]
    folder     = os.path.join(output_root, product_id)
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{timestamp.strftime('%Y%m%d_%H%M')}.npy")

    if os.path.exists(file_path):
        return {"status": "exists", "path": file_path, "product": product_id,
                "timestamp": timestamp.strftime("%Y%m%d_%H%M")}

    print(f"    📥 Descargando {timestamp.strftime('%Y%m%d_%H%M')} {product_id}...", end="", flush=True)
    start_time = time.time()

    try:
        band = product_cfg.get("band")

        if band in HIGH_RES_BANDS:
            # Flujo especial para bandas de alta resolución: abrir lazy y cropear
            data = download_highres(timestamp, product_cfg, region_cfg, satellite, domain)
        else:
            # Flujo normal: goes2go carga el dataset completo en memoria
            g  = GOES(satellite=satellite, product=product_cfg["product"], domain=domain,
                      bands=band if band else None) 
            ds = g.nearesttime(timestamp)
            xs, ys = goes_xy_slices(ds, **region_cfg)
            data = ds.sel(x=xs, y=ys)[product_cfg["variable"]].values
            ds.close()

        if data.size == 0:
            elapsed = time.time() - start_time
            print(f" ⚠️  vacío ({elapsed:.1f}s)")
            return {"status": "empty", "path": None, "product": product_id,
                    "timestamp": timestamp.strftime("%Y%m%d_%H%M")}

        dtype = np.float32 if product_cfg["dtype"] == "float32" else np.int8
        np.save(file_path, data.astype(dtype))

        elapsed = time.time() - start_time
        print(f" ✅ ({elapsed:.1f}s, shape={list(data.shape)})")
        return {"status": "downloaded", "path": file_path, "product": product_id,
                "timestamp": timestamp.strftime("%Y%m%d_%H%M"), "shape": list(data.shape)}

    except Exception as e:
        elapsed = time.time() - start_time
        print(f" ❌ ({elapsed:.1f}s)")

        # Definir s3_path antes del try para garantizar que siempre existe
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
