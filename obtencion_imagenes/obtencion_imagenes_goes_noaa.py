from goes2go import GOES
from datetime import datetime, timedelta
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pyproj

# ── Bounding box de Uruguay con margen ──────────────────────────────────────
# Uruguay está entre 30°S–35°S y 53°W–58°W (con un pequeño margen extra)
LAT_MIN, LAT_MAX = -35.5, -29.5
LON_MIN, LON_MAX = -59.0, -52.0

def goes_xy_from_latlon(ds, lat_min, lat_max, lon_min, lon_max):
    """
    Convierte un bounding box lat/lon a índices x/y del grid GOES.
    Devuelve (x_slice, y_slice) para usar con ds.sel().
    """
    # Leer parámetros de proyección desde el NetCDF
    proj_info = ds["goes_imager_projection"]
    lon_origin = float(proj_info.longitude_of_projection_origin)
    H          = float(proj_info.perspective_point_height) + float(proj_info.semi_major_axis)
    r_eq       = float(proj_info.semi_major_axis)
    r_pol      = float(proj_info.semi_minor_axis)

    # Construir proyección geoestacionaria con pyproj
    crs_goes = pyproj.CRS.from_dict({
        "proj": "geos",
        "lon_0": lon_origin,
        "h": float(proj_info.perspective_point_height),
        "a": r_eq,
        "b": r_pol,
        "sweep": "x",
    })
    crs_latlon = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs_latlon, crs_goes, always_xy=True)

    # Las 4 esquinas del bounding box → coordenadas de proyección (metros)
    corners_lon = [lon_min, lon_max, lon_min, lon_max]
    corners_lat = [lat_min, lat_min, lat_max, lat_max]
    px, py = transformer.transform(corners_lon, corners_lat)

    # El dataset GOES usa ángulos en radianes: x = px / H, y = py / H
    x_vals = ds["x"].values  # radianes, aumentan de oeste a este
    y_vals = ds["y"].values  # radianes, DECREMENTAN de norte a sur

    x_min_rad = np.nanmin(px) / H
    x_max_rad = np.nanmax(px) / H
    y_min_rad = np.nanmin(py) / H
    y_max_rad = np.nanmax(py) / H

    # Crear slices en las coordenadas del dataset
    x_slice = slice(
        float(x_vals[np.argmin(np.abs(x_vals - x_min_rad))]),
        float(x_vals[np.argmin(np.abs(x_vals - x_max_rad))]),
    )
    # y está invertido (mayor arriba), así que y_max va primero
    y_slice = slice(
        float(y_vals[np.argmin(np.abs(y_vals - y_max_rad))]),
        float(y_vals[np.argmin(np.abs(y_vals - y_min_rad))]),
    )
    return x_slice, y_slice


print("1. Configurando conexión con GOES-19...")
G = GOES(satellite=19, product="ABI-L1b-Rad", domain="F", bands=7)

print("2. Descargando la imagen...")
hora_segura = datetime.utcnow() - timedelta(hours=3)
print(f"   Buscando datos cercanos a las: {hora_segura.strftime('%H:%M')} UTC...")
ds = G.nearesttime(hora_segura)

print("3. Recortando al bounding box de Uruguay...")
x_slice, y_slice = goes_xy_from_latlon(ds, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
ds_uy = ds.sel(x=x_slice, y=y_slice)

print("4. Generando y guardando la imagen JPG...")
rad = ds_uy["Rad"].values

fig, ax = plt.subplots(figsize=(6, 8))
ax.axis("off")
plt.imshow(rad, cmap="inferno", vmin=np.nanpercentile(rad, 2), origin="upper")
plt.title(f"GOES-19 Banda 7 – Uruguay\n{hora_segura.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=10)

nombre_archivo = "goes19_uruguay.jpg"
plt.savefig(nombre_archivo, format="jpg", dpi=150, bbox_inches="tight")
print(f"¡Éxito! Imagen guardada como: {nombre_archivo}")
plt.show()