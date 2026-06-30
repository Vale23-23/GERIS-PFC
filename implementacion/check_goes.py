from goes2go import GOES
from datetime import datetime
import xarray as xr

# Solo descarga, no intenta abrir
g = GOES(satellite=19, product='ABI-L1b-Rad', domain='F', bands=2)
files = g.nearesttime(datetime(2025, 9, 26, 19, 0), return_as="filelist")
print("Archivo:", files)

# Abrir lazy (sin cargar en memoria)
import os
relative_path = files.iloc[0]["file"]
# goes2go guarda en C:\Users\luciana\data\ por defecto
full_path = os.path.join(r"C:\Users\luciana\data", relative_path)
print("Path completo:", full_path)
path = full_path  # el path local del .nc descargado
ds = xr.open_dataset(path, engine="h5netcdf")
print(ds)
print(ds.data_vars)

import pyproj
import numpy as np

# Coordenadas de Uruguay del config
lat_min, lat_max = -35.5, -29.5
lon_min, lon_max = -59.0, -52.0

# Misma función que usa downloader.py
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

cropped = ds.sel(x=x_slice, y=y_slice)["Rad"].values
print("Shape del crop:", cropped.shape)
print("Min/Max:", cropped.min(), cropped.max())

print("Unidades Rad:", ds["Rad"].attrs.get("units", "no definido"))
print("scale_factor:", ds["Rad"].attrs.get("scale_factor", "no definido"))
print("add_offset:",   ds["Rad"].attrs.get("add_offset",   "no definido"))

# Abrir B07 del npy guardado
rad7 = np.load(r"C:\Users\luciana\Desktop\Fing\PFC\GERIS-PFC\obtencion_imagenes\dataset\uruguay\ABI-L1b-Rad-B07\20250926_1900.npy")
print("B07 npy range:", rad7.min(), rad7.max())

# Aplicar conversión y Planck manualmente
from fdca.planck import planck_temp  # corré esto desde implementacion/
rad7_si = rad7 * 1e6
bt7 = planck_temp(7, rad7_si)
print("BT7 con *1e6:", np.nanmin(bt7), np.nanmax(bt7))

rad7_si2 = rad7 * 1e3  
bt7b = planck_temp(7, rad7_si2)
print("BT7 con *1e3:", np.nanmin(bt7b), np.nanmax(bt7b))

# Sin conversión
bt7c = planck_temp(7, rad7)
print("BT7 sin conv:", np.nanmin(bt7c), np.nanmax(bt7c))

rad14 = np.load(r"C:\Users\luciana\Desktop\Fing\PFC\GERIS-PFC\obtencion_imagenes\dataset\uruguay\ABI-L1b-Rad-B14\20250926_1900.npy")
print("B14 npy range:", rad14.min(), rad14.max())

rad14_si = rad14 * 1e6
bt14 = planck_temp(14, rad14_si)
print("BT14 con *1e6:", np.nanmin(bt14), np.nanmax(bt14))

g_ = GOES(satellite=19, product='ABI-L1b-Rad', domain='F', bands=14)
files_df = g_.nearesttime(datetime(2025, 9, 26, 19, 0), return_as="filelist")
full_path = os.path.join(r"C:\Users\luciana\data", files_df.iloc[0]["file"])
ds = xr.open_dataset(full_path, engine="h5netcdf")

rad = ds["Rad"]
print("Unidades B14:", rad.attrs.get("units"))
print("scale_factor:", rad.attrs.get("scale_factor", "no definido"))
print("add_offset:",   rad.attrs.get("add_offset",   "no definido"))
print("valid_range:",  rad.attrs.get("valid_range",  "no definido"))
print("Min/Max raw:",  float(rad.min()), float(rad.max()))
ds.close()