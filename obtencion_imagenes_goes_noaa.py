"""
Descarga de imágenes GOES-19 (GOES-East) - Banda 7 (3.9 µm)
Usando la librería goes2go desde el bucket público de AWS (NOAA Open Data)

Instalación:
    pip install goes2go xarray matplotlib netcdf4
"""

from goes2go import GOES
from datetime import datetime, timedelta
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# 1. CONFIGURAR EL OBJETO GOES
# ─────────────────────────────────────────────
# satellite: 19 = GOES-East (reemplaza al 16 desde principios de 2025)
# product:   ABI-L1b-Rad = datos radiométricos nivel 1b (crudos, en counts/radiancia)
#            ABI-L2-CMIP = producto de nube e imagen multicanal (nivel 2, procesado)
# domain:    'F' = Full Disk (disco completo, cubre Sudamérica)
#            'C' = CONUS (solo EEUU continental)
#            'M' = Mesoscale (zona de alta frecuencia)

G = GOES(
    satellite=19,
    product="ABI-L1b-Rad",   # nivel 1b para tener control total
    domain="F",               # Full Disk para ver Uruguay y región
)

# ─────────────────────────────────────────────
# 2. OPCIONES DE DESCARGA
# ─────────────────────────────────────────────

# --- Opción A: imagen más reciente disponible ---
ds = G.latest(bands=7)

# --- Opción B: imagen más cercana a una fecha/hora específica ---
# ds = G.nearesttime(
#     attime="2024-01-15 18:00",   # UTC
#     bands=7,                      # Banda 7 = infrarrojo de onda corta (3.9 µm)
# )

# --- Opción C: rango de tiempo (devuelve lista de archivos) ---
# ds = G.timerange(
#     start="2024-01-15 12:00",
#     end="2024-01-15 18:00",
#     bands=7,
# )

# ─────────────────────────────────────────────
# 3. EXPLORAR EL DATASET
# ─────────────────────────────────────────────
print(ds)
# El dataset xarray tiene:
#   ds["Rad"]   → radiancia en W·m⁻²·sr⁻¹·µm⁻¹
#   ds["DQF"]   → Data Quality Flag (0 = bueno)
#   ds.attrs    → metadatos del archivo NetCDF

# Convertir radiancia a temperatura de brillo (Brightness Temperature)
# Solo válido para bandas IR (bandas 7 al 16)
fk1 = ds["planck_fk1"].values
fk2 = ds["planck_fk2"].values
bc1 = ds["planck_bc1"].values
bc2 = ds["planck_bc2"].values

rad = ds["Rad"].values
BT = (fk2 / (np.log((fk1 / rad) + 1)) - bc1) / bc2  # en Kelvin

print(f"Temperatura de brillo mínima: {np.nanmin(BT):.1f} K")
print(f"Temperatura de brillo máxima: {np.nanmax(BT):.1f} K")

# ─────────────────────────────────────────────
# 4. VISUALIZACIÓN BÁSICA
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))

# Banda 7: focos ígneos aparecen como píxeles muy calientes (alta BT)
# Invertimos el colormap: áreas calientes = colores brillantes
im = ax.imshow(
    BT,
    cmap="inferno",
    vmin=200,   # ~-73°C (nubes frías)
    vmax=330,   # ~57°C (superficie muy caliente / fuego)
    origin="upper",
)

plt.colorbar(im, ax=ax, label="Temperatura de brillo (K)")
ax.set_title(f"GOES-19 | Banda 7 (3.9 µm) | {ds.attrs.get('time_coverage_start', '')}")
ax.axis("off")
plt.tight_layout()
plt.savefig("goes19_banda7.png", dpi=150)
plt.show()
print("Imagen guardada: goes19_banda7.png")

# ─────────────────────────────────────────────
# 5. DETECCIÓN SIMPLE DE FOCOS (umbral)
# ─────────────────────────────────────────────
# Umbral conservador para detección de fuego en Banda 7:
# píxeles con BT > 320 K (~47°C) son candidatos a focos ígneos

UMBRAL_FUEGO_K = 320.0
focos = BT > UMBRAL_FUEGO_K

n_focos = np.sum(focos)
print(f"Píxeles candidatos a foco ígneo (BT > {UMBRAL_FUEGO_K} K): {n_focos}")

# Coordenadas de los focos (en píxeles del grid)
y_focos, x_focos = np.where(focos)
print(f"Primeros 5 focos (fila, col): {list(zip(y_focos[:5], x_focos[:5]))}")

# ─────────────────────────────────────────────
# NOTAS PARA EL PROYECTO
# ─────────────────────────────────────────────
# - GOES-19 reemplazó a GOES-16 como GOES-East en 2025
#   goes2go todavía puede referenciarlo como satellite=16 en versiones viejas,
#   verificar con: G = GOES(satellite="EAST", ...)
#
# - Los archivos se descargan por defecto a ~/data/noaa-goes19/
#   Configurable con: G = GOES(..., save_dir="/ruta/custom")
#
# - Para Sudamérica con Full Disk, el grid es grande (~10k x 10k px)
#   Considerar recortar con ds.sel() usando coordenadas x/y del grid geoestacionario
#
# - Producto nivel 2 de focos ya procesado por NOAA:
#   product="ABI-L2-FDC"  ← Fire Detection and Characterization (FDC)
#   Útil para comparar contra el modelo propio