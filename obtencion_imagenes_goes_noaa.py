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
print("=" * 55)
print("  GOES-19 | Banda 7 (3.9 µm) | Descarga y análisis")
print("=" * 55)

print("\n[1/5] Configurando conexión con AWS (NOAA Open Data)...")
print("      Satélite : GOES-19 (GOES-East)")
print("      Producto : ABI-L1b-RadC07 (Infrarrojo onda corta)")
print("      Dominio  : Full Disk (cubre Sudamérica)")

G = GOES(
    satellite=19,
    product="ABI-L1b-Rad",  # banda en el nombre — fix para versiones antiguas
    domain="F",
)
print("      OK — objeto GOES listo")

# ─────────────────────────────────────────────
# 2. DESCARGA
# ─────────────────────────────────────────────
print("\n[2/5] Descargando imagen más reciente desde S3...")
print("      (puede tardar unos segundos según conexión)")

ds = G.latest()

timestamp = ds.attrs.get("time_coverage_start", "desconocido")
print(f"      OK — archivo descargado")
print(f"      Hora de captura (UTC) : {timestamp}")
print(f"      Dimensiones del grid  : {ds.dims['y']} filas x {ds.dims['x']} columnas")
print(f"      Variables disponibles : {list(ds.data_vars)}")

# ─────────────────────────────────────────────
# 3. CONVERSIÓN: RADIANCIA → TEMPERATURA DE BRILLO
# ─────────────────────────────────────────────
print("\n[3/5] Convirtiendo radiancia a temperatura de brillo (BT)...")
print("      Leyendo coeficientes de Planck del NetCDF...")

fk1 = float(ds["planck_fk1"])
fk2 = float(ds["planck_fk2"])
bc1 = float(ds["planck_bc1"])
bc2 = float(ds["planck_bc2"])

print(f"      fk1={fk1:.2f}  fk2={fk2:.2f}  bc1={bc1:.6f}  bc2={bc2:.6f}")
print("      Aplicando fórmula inversa de Planck...")

rad = ds["Rad"].values.astype(float)
rad = np.where(rad > 0, rad, np.nan)
BT  = (fk2 / (np.log((fk1 / rad) + 1)) - bc1) / bc2  # Kelvin

n_validos = int(np.sum(~np.isnan(BT)))
n_total   = BT.size
print(f"      OK — BT calculado")
print(f"      Píxeles válidos        : {n_validos:,} / {n_total:,}")
print(f"      BT mínima              : {np.nanmin(BT):.1f} K  ({np.nanmin(BT)-273.15:.1f} °C)")
print(f"      BT máxima              : {np.nanmax(BT):.1f} K  ({np.nanmax(BT)-273.15:.1f} °C)")
print(f"      BT media               : {np.nanmean(BT):.1f} K  ({np.nanmean(BT)-273.15:.1f} °C)")

# ─────────────────────────────────────────────
# 4. DETECCIÓN SIMPLE DE FOCOS (umbral)
# ─────────────────────────────────────────────
print("\n[4/5] Detectando focos ígneos candidatos...")

UMBRAL_FUEGO_K = 320.0
print(f"      Umbral aplicado : BT > {UMBRAL_FUEGO_K} K ({UMBRAL_FUEGO_K-273.15:.1f} °C)")

focos = BT > UMBRAL_FUEGO_K
n_focos = int(np.sum(focos & ~np.isnan(BT)))
pct = 100 * n_focos / n_validos

print(f"      Focos candidatos: {n_focos:,} px ({pct:.4f}% del área válida)")

if n_focos > 0:
    y_focos, x_focos = np.where(focos)
    bt_focos = BT[focos]
    print(f"      BT promedio en focos : {bt_focos.mean():.1f} K ({bt_focos.mean()-273.15:.1f} °C)")
    print(f"      BT máxima en focos   : {bt_focos.max():.1f} K ({bt_focos.max()-273.15:.1f} °C)")
    print(f"      Primeros 5 (fila, col): {list(zip(y_focos[:5].tolist(), x_focos[:5].tolist()))}")
else:
    print("      Sin focos detectados en esta imagen.")

# ─────────────────────────────────────────────
# 5. VISUALIZACIÓN
# ─────────────────────────────────────────────
print("\n[5/5] Generando visualización...")

fig, ax = plt.subplots(figsize=(10, 10))

im = ax.imshow(
    BT,
    cmap="inferno",
    vmin=200,
    vmax=330,
    origin="upper",
)

plt.colorbar(im, ax=ax, label="Temperatura de brillo (K)")
ax.set_title(
    f"GOES-19 | Banda 7 (3.9 µm) | {timestamp[:16].replace('T',' ')} UTC\n"
    f"Focos candidatos (BT > {UMBRAL_FUEGO_K} K): {n_focos:,} px"
)
ax.axis("off")
plt.tight_layout()

output_png = "goes19_banda7.png"
plt.savefig(output_png, dpi=150)
plt.show()
print(f"      OK — imagen guardada: {output_png}")

print("\n" + "=" * 55)
print("  Listo.")
print("=" * 55)