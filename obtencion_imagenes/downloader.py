"""
downloader.py — config-driven download and spatial crop for GOES data.
"""
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


def extract_units_metadata(da) -> dict:
    """
    Extrae la metadata de unidades de un xr.DataArray tal como viene del .nc
    de NOAA, para dejarla documentada en un sidecar JSON.

    Por qué hace falta: np.save() guarda solo el array de números, sin
    ningún atributo. Una vez escrito el .npy, la unidad física de esos
    números (¿mW m-2 sr-1 cm-1? ¿W m-2 sr-1 um-1? ¿K? ¿mm?) deja de estar
    en ningún lado salvo en la memoria de quien escribió el código. Esta
    función la captura en el momento en que todavía está disponible (el
    .nc recién abierto) y la persiste al lado del .npy correspondiente.

    - "units"/"long_name"/"valid_range" vienen de da.attrs, que es donde
      xarray deja los metadatos ya decodificados (CF-compliant).
    - "scale_factor"/"add_offset" viven en da.encoding, no en da.attrs:
      xarray los saca de attrs y los mueve ahí en el momento en que
      decodifica el entero empaquetado a su valor físico. Si aparecen acá
      es justamente la prueba de que da.values YA está en unidades físicas
      (no en el entero crudo sin escalar) — que es el supuesto que usa
      todo fdca_adapter.py.
    """
    attrs = da.attrs
    enc   = da.encoding

    valid_range = attrs.get("valid_range")
    if valid_range is not None:
        valid_range = [float(x) for x in np.asarray(valid_range).tolist()]

    cf_decoded = ("scale_factor" in enc) or ("add_offset" in enc)

    return {
        "units":        attrs.get("units"),
        "long_name":    attrs.get("long_name"),
        "valid_range":  valid_range,
        "cf_decoded":   cf_decoded,     # True → .values ya está en unidades físicas
        "scale_factor": enc.get("scale_factor"),
        "add_offset":   enc.get("add_offset"),
        "packed_dtype": str(enc.get("dtype")) if "dtype" in enc else None,
    }


def download_highres(timestamp, product_cfg, region_cfg, satellite, domain, folder=None):
    """
    Descarga y cropea bandas de alta resolución (ej: B02).
    No guarda geometría propia: B02 termina remuestreado a la grilla de 2 km
    de las bandas IR en fdca_adapter.py, así que la única geometría que
    importa para el FDCAInput es region_geometry.json (generado por las
    bandas IR en la rama normal de download_and_save).

    Returns
    -------
    data  : np.ndarray
    units_meta : dict   (ver extract_units_metadata)
    """
    import xarray as xr
    band = product_cfg.get("band")
    g = GOES(satellite=satellite, product=product_cfg["product"], domain=domain, bands=band)
    files_df = g.nearesttime(timestamp, return_as="filelist")
    if files_df is None or len(files_df) == 0:
        raise RuntimeError(f"goes2go no encontró archivos para {product_cfg['id']} en {timestamp}")
    relative_path = files_df.iloc[0]["file"]
    cache_dir = get_goes2go_cache_dir()
    full_path = os.path.join(cache_dir, relative_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Archivo descargado no encontrado en: {full_path}")

    ds = xr.open_dataset(full_path)   # sin chunks, lazy por indexing igual
    xs, ys = goes_xy_slices(ds, **region_cfg)
    da = ds.sel(x=xs, y=ys)[product_cfg["variable"]]
    units_meta = extract_units_metadata(da)
    data = da.values
    ds.close()
    return data, units_meta

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
    units_path  = os.path.join(folder, f"{timestamp.strftime('%Y%m%d_%H%M')}_units.json")
    dqf_path    = os.path.join(folder, f"{timestamp.strftime('%Y%m%d_%H%M')}_dqf.npy")

    band = product_cfg.get("band")

    # ¿Hace falta descargar la imagen?
    need_npy = not os.path.exists(file_path)

    # ¿Hace falta generar el JSON con los coeficientes Planck?
    need_json = (
        band in PLANCK_BANDS
        and not os.path.exists(coeffs_path)
    )

    # ¿Hace falta generar el JSON con la metadata de unidades? A diferencia
    # de los coeficientes Planck, esto se guarda para TODOS los productos
    # (B02, DQF, TPW, etc.), no solo las bandas IR.
    need_units = not os.path.exists(units_path)

    # ¿Hace falta generar el .npy con el Data Quality Flag?
    need_dqf  = band == 7 and not os.path.exists(dqf_path)

    # Si ya existe todo lo necesario, no hacemos nada.
    if not need_npy and not need_json and not need_units:
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
            data, units_meta = download_highres(timestamp, product_cfg, region_cfg, satellite, domain, folder)
            planck_coeffs = None # Si el .npy existe pero falta el JSON de coeficientes (banda IR), lo señalamos
        else:
            g  = GOES(satellite=satellite, product=product_cfg["product"], domain=domain,
                    bands=band if band else None)
            ds = g.nearesttime(timestamp)
            xs, ys = goes_xy_slices(ds, **region_cfg)
            ds_cropped = ds.sel(x=xs, y=ys)
            da = ds_cropped[product_cfg["variable"]]
            units_meta = extract_units_metadata(da)
            data = da.values

            # Geometría: una sola vez por producto/banda (la grilla x/y es estática
            # en el tiempo; lo único que cambia por resolución es el band/product_id,
            geom_path = os.path.join(os.path.dirname(folder), "geometry.json")
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
                    "planck_units": {
                        "fk1": ds["planck_fk1"].attrs.get("units"),
                        "fk2": ds["planck_fk2"].attrs.get("units"),
                        "bc1": ds["planck_bc1"].attrs.get("units"),
                        "bc2": ds["planck_bc2"].attrs.get("units"),
                    },
                }
                # FPT real (ATBD 3.4.2.2): el L1b no trae la temperatura literal,
                # sino un contador de QC que indica si el umbral de 90 K fue
                # superado en ese escaneo (tratable como booleano).
                if "focal_plane_temperature_threshold_exceeded_count" in ds.variables:
                    planck_coeffs["fpt_threshold_exceeded_count"] = int(
                        ds["focal_plane_temperature_threshold_exceeded_count"].values
                    )

            # DQF por pixel, recortado a la región (misma grilla que la radiancia).
            # DQF==4 marca los pixeles con focal_plane_temperature_threshold_exceeded_qf.
            if need_dqf and "DQF" in ds_cropped.variables:
                np.save(dqf_path, ds_cropped["DQF"].values.astype(np.int8))
                
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

        # Guardar el JSON de coeficientes Planck solo si hacía falta
        if need_json and planck_coeffs is not None:
            with open(coeffs_path, "w") as f:
                json.dump(planck_coeffs, f)

        # Guardar el JSON de unidades — para TODOS los productos, siempre que
        # falte. Es la única forma de saber, más adelante, en qué unidad
        # física quedó guardado un .npy que ya no tiene ningún atributo.
        if need_units:
            units_record = {
                "product_id": product_id,
                "variable":   product_cfg["variable"],
                "band":       band,
                **units_meta,
            }
            with open(units_path, "w") as f:
                json.dump(units_record, f, indent=2)

        elapsed = time.time() - start_time
        print(f" ✅ ({elapsed:.1f}s, shape={list(data.shape)}, units={units_meta.get('units')})")
        return {"status": "downloaded", "path": file_path, "product": product_id,
                "timestamp": timestamp.strftime("%Y%m%d_%H%M"), "shape": list(data.shape),
                "units": units_meta.get("units")}

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

# import os
# import numpy as np
# import pyproj
# from goes2go import GOES

# # Bandas de alta resolución que no pueden cargarse con nearesttime() completo
# # porque el full disk pesa demasiado (21696x21696 píxeles)
# HIGH_RES_BANDS = {2}


# def goes_xy_slices(ds, lat_min, lat_max, lon_min, lon_max):
#     """Convert a lat/lon bounding box to GOES x/y slices."""
#     proj_info  = ds["goes_imager_projection"]
#     lon_origin = float(proj_info.longitude_of_projection_origin)
#     H          = float(proj_info.perspective_point_height) + float(proj_info.semi_major_axis)

#     crs_goes = pyproj.CRS.from_dict({
#         "proj": "geos", "lon_0": lon_origin,
#         "h": float(proj_info.perspective_point_height),
#         "a": float(proj_info.semi_major_axis),
#         "b": float(proj_info.semi_minor_axis),
#         "sweep": "x",
#     })
#     transformer = pyproj.Transformer.from_crs(
#         pyproj.CRS.from_epsg(4326), crs_goes, always_xy=True
#     )
#     px, py = transformer.transform(
#         [lon_min, lon_max, lon_min, lon_max],
#         [lat_min, lat_min, lat_max, lat_max],
#     )
#     x_vals, y_vals = ds["x"].values, ds["y"].values
#     x_slice = slice(
#         float(x_vals[np.argmin(np.abs(x_vals - np.nanmin(px) / H))]),
#         float(x_vals[np.argmin(np.abs(x_vals - np.nanmax(px) / H))]),
#     )
#     y_slice = slice(
#         float(y_vals[np.argmin(np.abs(y_vals - np.nanmax(py) / H))]),
#         float(y_vals[np.argmin(np.abs(y_vals - np.nanmin(py) / H))]),
#     )
#     return x_slice, y_slice


# def get_goes2go_cache_dir():
#     """Devuelve el directorio donde goes2go guarda los archivos descargados."""
#     try:
#         from goes2go.config import config as g2g_config
#         return g2g_config.get("default", {}).get("save_dir", os.path.expanduser("~/data"))
#     except Exception:
#         return os.path.expanduser("~/data")

# def download_highres(timestamp, product_cfg, region_cfg, satellite, domain, folder=None):
#     """
#     Descarga y cropea bandas de alta resolución (ej: B02).
#     No guarda geometría propia: B02 termina remuestreado a la grilla de 2 km
#     de las bandas IR en fdca_adapter.py, así que la única geometría que
#     importa para el FDCAInput es region_geometry.json (generado por las
#     bandas IR en la rama normal de download_and_save).
#     """
#     import xarray as xr
#     band = product_cfg.get("band")
#     g = GOES(satellite=satellite, product=product_cfg["product"], domain=domain, bands=band)
#     files_df = g.nearesttime(timestamp, return_as="filelist")
#     if files_df is None or len(files_df) == 0:
#         raise RuntimeError(f"goes2go no encontró archivos para {product_cfg['id']} en {timestamp}")
#     relative_path = files_df.iloc[0]["file"]
#     cache_dir = get_goes2go_cache_dir()
#     full_path = os.path.join(cache_dir, relative_path)
#     if not os.path.exists(full_path):
#         raise FileNotFoundError(f"Archivo descargado no encontrado en: {full_path}")

#     ds = xr.open_dataset(full_path)   # sin chunks, lazy por indexing igual
#     xs, ys = goes_xy_slices(ds, **region_cfg)
#     data = ds.sel(x=xs, y=ys)[product_cfg["variable"]].values
#     ds.close()
#     return data

# def download_and_save(timestamp, product_cfg, region_cfg, satellite, domain, output_root):
#     """
#     Download a single product/timestamp, crop to region, save as .npy.
#     For IR bands (7, 13, 14, 15), also saves Planck calibration coefficients
#     as a parallel JSON file.

#     Returns a dict with status and metadata, never raises.
#     """
#     import time
#     import json

#     PLANCK_BANDS = {7, 13, 14, 15} # Bandas IR donde nos importa la temperatura de brillo

#     product_id = product_cfg["id"]
#     folder     = os.path.join(output_root, product_id)
#     os.makedirs(folder, exist_ok=True)

#     file_path   = os.path.join(folder, f"{timestamp.strftime('%Y%m%d_%H%M')}.npy")
#     coeffs_path = os.path.join(folder, f"{timestamp.strftime('%Y%m%d_%H%M')}_planck.json")

#     band = product_cfg.get("band")

#     # ¿Hace falta descargar la imagen?
#     need_npy = not os.path.exists(file_path)

#     # ¿Hace falta generar el JSON con los coeficientes Planck?
#     need_json = (
#         band in PLANCK_BANDS
#         and not os.path.exists(coeffs_path)
#     )

#     # Si ya existe todo lo necesario, no hacemos nada.
#     if not need_npy and not need_json:
#         return {
#             "status": "exists",
#             "path": file_path,
#             "product": product_id,
#             "timestamp": timestamp.strftime("%Y%m%d_%H%M"),
#         }

#     print(f"    📥 Descargando {timestamp.strftime('%Y%m%d_%H%M')} {product_id}...", end="", flush=True)
#     start_time = time.time()

#     try:
#         if band in HIGH_RES_BANDS:
#             data = download_highres(timestamp, product_cfg, region_cfg, satellite, domain, folder)
#             planck_coeffs = None # Si el .npy existe pero falta el JSON de coeficientes (banda IR), lo señalamos
#         else:
#             g  = GOES(satellite=satellite, product=product_cfg["product"], domain=domain,
#                     bands=band if band else None)
#             ds = g.nearesttime(timestamp)
#             xs, ys = goes_xy_slices(ds, **region_cfg)
#             ds_cropped = ds.sel(x=xs, y=ys)
#             data = ds_cropped[product_cfg["variable"]].values

#             # Geometría: una sola vez por producto/banda (la grilla x/y es estática
#             # en el tiempo; lo único que cambia por resolución es el band/product_id,
#             geom_path = os.path.join(os.path.dirname(folder), "geometry.json")
#             if not os.path.exists(geom_path):
#                 proj_info = ds_cropped["goes_imager_projection"]
#                 geom_meta = {
#                     "x": ds_cropped["x"].values.tolist(),
#                     "y": ds_cropped["y"].values.tolist(),
#                     "longitude_of_projection_origin": float(proj_info.longitude_of_projection_origin),
#                     "perspective_point_height": float(proj_info.perspective_point_height),
#                     "semi_major_axis": float(proj_info.semi_major_axis),
#                     "semi_minor_axis": float(proj_info.semi_minor_axis),
#                     "sweep_angle_axis": "x",
#                 }
#                 with open(geom_path, "w") as f:
#                     json.dump(geom_meta, f)

#             planck_coeffs = None
#             if band in PLANCK_BANDS:
#                 planck_coeffs = {
#                     "planck_fk1": float(ds["planck_fk1"].values),
#                     "planck_fk2": float(ds["planck_fk2"].values),
#                     "planck_bc1": float(ds["planck_bc1"].values),
#                     "planck_bc2": float(ds["planck_bc2"].values),
#                 }

#             ds.close()

#         if data.size == 0:
#             elapsed = time.time() - start_time
#             print(f" ⚠️  vacío ({elapsed:.1f}s)")
#             return {"status": "empty", "path": None, "product": product_id,
#                     "timestamp": timestamp.strftime("%Y%m%d_%H%M")}

#         # Guardar la imagen solo si hacía falta descargarla
#         if need_npy:
#             dtype = np.float32 if product_cfg["dtype"] == "float32" else np.int8
#             np.save(file_path, data.astype(dtype))

#         # Guardar el JSON solo si hacía falta
#         if need_json and planck_coeffs is not None:
#             with open(coeffs_path, "w") as f:
#                 json.dump(planck_coeffs, f)

#         elapsed = time.time() - start_time
#         print(f" ✅ ({elapsed:.1f}s, shape={list(data.shape)})")
#         return {"status": "downloaded", "path": file_path, "product": product_id,
#                 "timestamp": timestamp.strftime("%Y%m%d_%H%M"), "shape": list(data.shape)}

#     except Exception as e:
#         elapsed = time.time() - start_time
#         print(f" ❌ ({elapsed:.1f}s)")

#         s3_path    = None
#         status_msg = "error_local"
#         error_detail = f"Error descargando ({elapsed:.1f}s): {str(e)}"

#         try:
#             exists_on_s3, s3_path = check_s3_exists(product_cfg, timestamp)
#             if not exists_on_s3:
#                 status_msg   = "error_aws_gap"
#                 error_detail = f"Data Gap en AWS: no hay datos para {product_id} en s3://{s3_path}"
#             else:
#                 status_msg   = "error_local"
#                 error_detail = f"El archivo existe en S3 pero falló la descarga ({elapsed:.1f}s): {str(e)}"
#         except Exception as s3_err:
#             error_detail += f" | S3 check también falló: {str(s3_err)}"

#         return {
#             "status":     "error",
#             "substatus":  status_msg,
#             "product":    product_id,
#             "timestamp":  timestamp.strftime("%Y%m%d_%H%M"),
#             "error":      error_detail,
#             "product_path": s3_path,
#         }