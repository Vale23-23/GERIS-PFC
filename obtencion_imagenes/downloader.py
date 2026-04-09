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
        import traceback
        traceback.print_exc()
        return {"status": "error", "path": None, "product": product_id,
                "timestamp": timestamp.strftime("%Y%m%d_%H%M"), "error": str(e)}
