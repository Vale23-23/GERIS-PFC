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
    """Verifica si existen archivos en S3 para cualquier producto configurado."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    
    # Extraemos los datos del timestamp
    julian_day = timestamp.strftime('%j')
    hour = timestamp.strftime('%H')
    year = timestamp.strftime('%Y')
    
    # El 'product' en AWS suele ser el prefijo del ID (ej: ABI-L1b-Rad)
    # pero sin la banda específica. Lo tomamos directamente del config.
    s3_product = product_cfg["product"] 
    
    # Ajuste para productos Full Disk (RadF)
    if s3_product == "ABI-L1b-Rad":
        s3_product = "ABI-L1b-RadF"
        
    path = f"noaa-goes19/{s3_product}/{year}/{julian_day}/{hour}/"
    
    try:
        files = fs.ls(path)
        # Filtramos para asegurarnos que haya archivos de la banda correcta si es L1b
        if "band" in product_cfg:
            band_str = f"M6C{product_cfg['band']:02d}"
            files = [f for f in files if band_str in f]
            
        return len(files) > 0, path
    except:
        return False, path


def download_and_save(timestamp, product_cfg, region_cfg, satellite, domain, output_root):
    """
    Download a single product/timestamp, crop to region, save as .npy.

    Returns a dict with status and metadata, never raises.
    """
    product_id = product_cfg["id"]
    folder     = os.path.join(output_root, product_id)
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{timestamp.strftime('%Y%m%d_%H%M')}.npy")

    if os.path.exists(file_path):
        return {"status": "exists", "path": file_path, "product": product_id,
                "timestamp": timestamp.strftime("%Y%m%d_%H%M")}

    try:
        band = product_cfg.get("band")
        g    = GOES(satellite=satellite, product=product_cfg["product"], domain=domain,
                    bands=band if band else None)
        ds   = g.nearesttime(timestamp)

        xs, ys = goes_xy_slices(ds, **region_cfg)
        data   = ds.sel(x=xs, y=ys)[product_cfg["variable"]].values

        if data.size == 0:
            return {"status": "empty", "path": None, "product": product_id,
                    "timestamp": timestamp.strftime("%Y%m%d_%H%M")}

        dtype  = np.float32 if product_cfg["dtype"] == "float32" else np.int8
        np.save(file_path, data.astype(dtype))

        return {"status": "downloaded", "path": file_path, "product": product_id,
                "timestamp": timestamp.strftime("%Y%m%d_%H%M"), "shape": list(data.shape)}

    except Exception as e:
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
            "error": error_detail
        }
