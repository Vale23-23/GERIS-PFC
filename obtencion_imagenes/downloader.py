"""
downloader.py — config-driven download and spatial crop for GOES data.
"""

import os
import numpy as np
import pyproj
from goes2go import GOES


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

def check_s3_exists(product_cfg, timestamp):
    """Verifica existencia en S3 usando boto3 (más rápido y confiable que s3fs)."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    julian_day = timestamp.strftime('%j')
    hour = timestamp.strftime('%H')
    year = timestamp.strftime('%Y')
    
    s3_product = product_cfg["product"] 
    if s3_product == "ABI-L1b-Rad":
        s3_product = "ABI-L1b-RadF"
        
    prefix = f"{s3_product}/{year}/{julian_day}/{hour}/"
    bucket = "noaa-goes19" # O el que estés usando

    try:
        # Configuración de timeout estricta
        s3 = boto3.client('s3', config=Config(connect_timeout=2, read_timeout=2, retries={'max_attempts': 0}), region_name='us-east-1')
        
        # Listar solo 1 objeto para verificar existencia
        response = s3.list_objects_v2(
            Bucket=bucket, 
            Prefix=prefix, 
            MaxKeys=5, 
            RequestPayer='requester' # A veces necesario, o usar Config(signature_version=UNSIGNED)
        )
        
        files = response.get('Contents', [])
        if "band" in product_cfg:
            band_str = f"M6C{product_cfg['band']:02d}"
            exists = any(band_str in f['Key'] for f in files)
        else:
            exists = len(files) > 0
            
        return exists, f"{bucket}/{prefix}"
    except:
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

    try:
        print(f"    📥 Descargando {timestamp.strftime('%Y%m%d_%H%M')} {product_id}...", end="", flush=True)
        start_time = time.time()
        
        band = product_cfg.get("band")
        g    = GOES(satellite=satellite, product=product_cfg["product"], domain=domain,
                    bands=band if band else None, timeout=30)  # Timeout directo en GOES
        ds   = g.nearesttime(timestamp)

        xs, ys = goes_xy_slices(ds, **region_cfg)
        data   = ds.sel(x=xs, y=ys)[product_cfg["variable"]].values

        ds.close()

        if data.size == 0:
            print(f" ⚠️  vacío ({time.time()-start_time:.1f}s)")
            return {"status": "empty", "path": None, "product": product_id,
                    "timestamp": timestamp.strftime("%Y%m%d_%H%M")}

        dtype  = np.float32 if product_cfg["dtype"] == "float32" else np.int8
        np.save(file_path, data.astype(dtype))

        elapsed = time.time() - start_time
        print(f" ✅ ({elapsed:.1f}s, shape={list(data.shape)})")
        return {"status": "downloaded", "path": file_path, "product": product_id,
                "timestamp": timestamp.strftime("%Y%m%d_%H%M"), "shape": list(data.shape)}

    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print(f" ❌ ({elapsed:.1f}s)")
        
        # Verificación S3 simplificada y más rápida
        try:
            exists_on_s3, s3_path = check_s3_exists(product_cfg, timestamp)
            if not exists_on_s3:
                status_msg = "error_aws_gap"
                error_detail = f"Data Gap en AWS: No hay archivos para {product_id} en s3://{s3_path}"
            else:
                status_msg = "error_local"
                error_detail = f"Error local ({elapsed:.1f}s): {str(e)}"
        except:
            # Si la verificación S3 también falla, asumir error local
            status_msg = "error_local"
            error_detail = f"Error desconocido ({elapsed:.1f}s): {str(e)}"

        return {
            "status": "error", 
            "substatus": status_msg,
            "product": product_id,
            "timestamp": timestamp.strftime("%Y%m%d_%H%M"), 
            "error": error_detail,
            "product_path": s3_path if 's3_path' in locals() else None
        }
    
        # Pasamos el config completo para que la verificación sea precisa por producto

        exists_on_s3, s3_path = check_s3_exists(product_cfg, timestamp)
        
        if not exists_on_s3:
            status_msg = "error_aws_gap"
            error_detail = f"Data Gap en AWS: No hay archivos para {product_id} en s3://{s3_path}"
        else:
            status_msg = "error_local"
            error_detail = f"Error local: El archivo existe en S3 pero falló el proceso. {str(e)}"

        # Imprimimos un mensaje claro en la terminal
        print(f"  ❌ {timestamp.strftime('%Y%m%d_%H%M')} -> {status_msg}")
        
        return {
            "status": "error", 
            "substatus": status_msg,
            "product": product_id,
            "timestamp": timestamp.strftime("%Y%m%d_%H%M"), 
            "error": error_detail,
            "product_path": s3_path  # Agregado para que pipeline.py pueda acceder
        }
