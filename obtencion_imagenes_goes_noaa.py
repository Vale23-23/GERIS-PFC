from goes2go import GOES
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

print("1. Configurando conexión con GOES-19...")
G = GOES(satellite=19, product="ABI-L1b-Rad", domain="F", bands=7)

print("2. Descargando la imagen...")
# SOLUCIÓN: Restamos 3 horas a la hora actual UTC
hora_segura = datetime.utcnow() - timedelta(hours=3)
print(f"   Buscando datos cercanos a las: {hora_segura.strftime('%H:%M')} UTC...")

# Busca y descarga el archivo NetCDF
ds = G.nearesttime(hora_segura)

print("3. Generando y guardando la imagen JPG...")
rad = ds["Rad"].values

fig, ax = plt.subplots(figsize=(8, 8))
ax.axis("off") 

plt.imshow(rad, cmap="inferno", vmin=0, origin="upper")

nombre_archivo = "goes19_simple.jpg"
plt.savefig(nombre_archivo, format="jpg", dpi=150, bbox_inches="tight")
print(f"¡Éxito! Imagen guardada en tu carpeta como: {nombre_archivo}")

plt.show()