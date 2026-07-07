"""
fdca_adapter.py
───────────────
Convierte los archivos .npy del pipeline de obtención de imágenes en un
FDCAInput listo para correr el algoritmo FDCA.

El pipeline ya descarga y recorta los datos; este módulo solo hace la
"traducción" entre el formato del pipeline y el formato que espera el FDCA.

=============================================================================
TABLA DE UNIDADES Y RANGOS DE REFERENCIA
=============================================================================
1. RADIANCIAS CRUDAS (ABI-L1b-Rad-BXX):
    ABI-L1b-Rad-B02: 
   - Unidad en archivos netCDF: W * m^-2 * sr^-1 * µm^-1
    ABI-L1b-Rad-B07, ABI-L1b-Rad-B13,ABI-L1b-Rad-B14, ABI-L1b-Rad-B15: 
   - Unidad en archivos netCDF: mW * m^-2 * sr^-1 * (cm-1)^-1
 

2. TEMPERATURAS DE BRILLO (BT7, BT13, BT14, BT15):
   - Unidad resultante: Kelvin (K)  [Conversión via Planck Inverso]


3. REFLECTANCIA (refl2):
   - Unidad: Adimensional / Factor de Reflectancia (Rango 0.0 a 1.0)
   - Cálculo: Radiancia B02 / Radiancia Solar Teórica (ajustada por SZA).
   - Rangos esperados: 0.02 a 0.35 (Suelo libre de nubes); >0.40 (Nubes/Glint).

4. GEOMETRÍA (SZA, LZA, Glint):
   - Unidad: Grados sexagesimales (°)
   - Rangos/Límites del Algoritmo:
     * SZA (Solar): < 85° para modo Diurno.
     * LZA (Satélite): < 80° (Límite estricto FDCA), óptimo < 65° (Uruguay: ~31°-41°).

5. TPW (Total Precipitable Water):
   - Unidad: Milímetros (mm)
   - Rangos climatológicos Cono Sur: 15 mm (Invierno seco) a 45 mm (Verano húmedo).
=============================================================================

Qué hace cada sección:
  1. Lee las radiancias de B07, B14 (y opcionalmente B13, B15, B02)
  2. Convierte radiancia → BT usando Planck inverso
  3. Convierte B02 (Rad) → reflectance factor (Rad/Rad_solar)
  4. Calcula geometría solar (SZA, LZA, glint_angle) a partir de lat/lon y hora
  5. Reconstruye lat/lon desde el config de región (grilla uniforme aproximada)
  6. Arma las máscaras de superficie (tierra/agua, land cover, etc.)
  7. Construye el LUT de corrección TPW
  8. Empaqueta todo en FDCAInput

Uso:
  from fdca_adapter import load_fdca_input
  inp = load_fdca_input("20250905_1500", region="uruguay")
  # → FDCAInput listo para run_fdca(inp)
"""

from __future__ import annotations  # permite 'np.ndarray | None' en Python < 3.10


import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import pyproj

# ── Constantes físicas para conversión de radiancia B02 ──────────────────────
# Irradiancia solar exoatmosférica para ABI Band 2 (0.64 µm) [W·m⁻²·µm⁻¹]
# Valor de la tabla de calibración ABI (GOES-R PUG-L1B, Tabla 7-6)
ESUN_B02 = 1622.088   # [W·m⁻²·µm⁻¹] #CHEQUEAR ESTE VALOR CUAL ES
# Factor de corrección: Rad [W·m⁻²·sr⁻¹·µm⁻¹] → Reflectance factor [-]
# refl_factor = π * Rad / (ESUN * cos(SZA))
# Acá calculamos sin dividir por cos(SZA) — eso lo hace part1.py al calcular albedo

# Constantes GOES-19 — SOLO fallback si geometry.json no trae estos campos.
# IMPORTANTE: compute_latlon_grid() y compute_local_zenith() deben usar los
# valores reales leídos de geometry.json (lon_0, h, a, b), no estas constantes,
# para que las coordenadas y los ángulos sean geométricamente consistentes
# entre sí. Antes este archivo calculaba lat/lon con el elipsoide real del
# JSON pero el LZA con una fórmula esférica + esta constante hardcodeada:
# quedaban desacopladas. Ver load_geometry().

GOES19_LON_0_FALLBACK  = -75.0          # longitud subsatelital [deg]
GOES19_H_FALLBACK      = 35786.023e3    # altura orbital [m]
GOES19_R_EQ_FALLBACK   = 6378.137e3     # radio ecuatorial [m]
GOES19_R_POL_FALLBACK  = 6356.7523e3    # radio polar [m]


def load_geometry(base_path: str) -> dict:
    """
    Única fuente de verdad para los parámetros geométricos del satélite.
    Lee geometry.json y devuelve lon_0, h, a, b (mismos parámetros que usa
    compute_latlon_grid para proyectar lat/lon). compute_local_zenith y el
    cálculo de azimuth DEBEN recibir estos mismos valores, para no volver a
    quedar desacoplados de las coordenadas.
    """
    geom_path = os.path.join(base_path, "geometry.json")
    if not os.path.exists(geom_path):
        raise FileNotFoundError(
            f"No se encontró la geometría de la región en: {geom_path}. "
            f"Asegúrate de correr el downloader para generar este archivo base."
        )
    with open(geom_path, "r") as f:
        geom_data = json.load(f)

    return {
        "lon_0": float(geom_data["longitude_of_projection_origin"]),
        "h":     float(geom_data["perspective_point_height"]),
        "a":     float(geom_data["semi_major_axis"]),
        "b":     float(geom_data["semi_minor_axis"]),
    }


# ── Geometría ─────────────────────────────────────────────────────────────────

def compute_latlon_grid(base_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye la grilla bidimensional de Latitud y Longitud exacta utilizando
    la elipsoide y la proyección geoestacionaria (Fixed Grid System) del ABI.
    Lee los metadatos geométricos fijos generados por el downloader.

    Parameters
    ----------
    base_path : str
        Ruta raíz de la región (ej. "dataset/uruguay") donde reside 'region_geometry.json'.

    Returns
    -------
    lat2d, lon2d : arrays [H, W] con coordenadas exactas o np.nan en espacio profundo.
    """
    import os
    import json
    import pyproj
    import numpy as np

    # 1. Cargar el JSON geométrico de la región (misma función que usan
    #    compute_local_zenith / compute_view_geometry más abajo, para que
    #    coordenadas y ángulos usen siempre el mismo lon_0/h/a/b)
    geom = load_geometry(base_path)

    geom_path = os.path.join(base_path, "geometry.json")
    with open(geom_path, "r") as f:
        geom_data = json.load(f)

    # Convertir listas del JSON a vectores numéricos de NumPy
    x_vals = np.array(geom_data["x"], dtype=np.float32)
    y_vals = np.array(geom_data["y"], dtype=np.float32)

    # 2. Configurar la Proyección Geoestacionaria (Fixed Grid) usando los datos del satélite
    crs_goes = pyproj.CRS.from_dict({
        "proj": "geos",
        "lon_0": geom["lon_0"],
        "h": geom["h"],
        "a": geom["a"],
        "b": geom["b"],
        "sweep": "x",
    })

    # 3. Configurar el sistema de destino (WGS84 estándar en grados)
    crs_latlon = pyproj.CRS.from_epsg(4326)

    # 4. Crear el transformador matemático (always_xy=True garantiza orden Long, Lat)
    transformer = pyproj.Transformer.from_crs(crs_goes, crs_latlon, always_xy=True)

    # 5. Proyectar la malla 2D
    # En pyproj, las coordenadas geostacionarias deben multiplicarse por la altura orbital (h)
    h = geom["h"]
    x_mesh, y_mesh = np.meshgrid(x_vals * h, y_vals * h)

    # Transformación de coordenadas de matriz completa
    lon2d, lat2d = transformer.transform(x_mesh, y_mesh)

    # 6. Control de bordes (Filtrar píxeles inválidos que caen fuera del disco de la Tierra)
    lon2d = np.where(np.abs(lon2d) <= 180, lon2d, np.nan)
    lat2d = np.where(np.abs(lat2d) <= 90, lat2d, np.nan)

    return lat2d.astype(np.float32), lon2d.astype(np.float32)


def _solar_position(lat: np.ndarray, lon: np.ndarray, dt: datetime):
    """
    Declinación solar y ángulo horario local — factorizado para que
    compute_solar_zenith y compute_solar_azimuth usen EXACTAMENTE los mismos
    dec/hour_angle.
    """
    doy = dt.timetuple().tm_yday
    hour_utc = dt.hour + dt.minute / 60.0

    B = np.radians((360.0 / 365.0) * (doy - 81))
    dec_rad = np.radians(23.45 * np.sin(B))

    # Ecuación del tiempo (corrección de minutos solares)
    eot = (9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)) / 60.0
    lstm = 0.0   # hour_utc ya está referido al meridiano de Greenwich
    tc   = 4.0 * (lon - lstm) + 60.0 * eot
    hour_local = hour_utc + tc / 60.0
    hour_angle_rad = np.radians(15.0 * (hour_local - 12.0))

    return dec_rad, hour_angle_rad


def compute_solar_zenith(lat: np.ndarray, lon: np.ndarray, dt: datetime) -> np.ndarray:
    """
    Ángulo cenital solar [deg] usando la aproximación de Spencer (±0.5°).
    Suficiente para los tests día/noche del FDCA.
    """
    dec_rad, hour_angle_rad = _solar_position(lat, lon, dt)
    lat_rad = np.radians(lat)
    cos_sza = (np.sin(lat_rad) * np.sin(dec_rad) +
               np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_angle_rad))
    cos_sza = np.clip(cos_sza, -1.0, 1.0)
    return np.degrees(np.arccos(cos_sza)).astype(np.float32)


def compute_solar_azimuth(lat: np.ndarray, lon: np.ndarray, dt: datetime) -> np.ndarray:
    """
    Azimuth solar [deg, desde el Norte, sentido horario].
    Usa la misma declinación/ángulo horario que compute_solar_zenith
    (vía _solar_position) para que ambos ángulos sean consistentes entre sí.
    Fórmula estándar (NOAA Solar Calculator).
    """
    dec_rad, hour_angle_rad = _solar_position(lat, lon, dt)
    lat_rad = np.radians(lat)

    cos_sza = (np.sin(lat_rad) * np.sin(dec_rad) +
               np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_angle_rad))
    cos_sza = np.clip(cos_sza, -1.0, 1.0)
    zenith_rad = np.arccos(cos_sza)
    sin_zenith_safe = np.where(np.sin(zenith_rad) > 1e-6, np.sin(zenith_rad), 1e-6)

    cos_az = ((np.sin(lat_rad) * np.cos(zenith_rad) - np.sin(dec_rad)) /
              (np.cos(lat_rad) * sin_zenith_safe))
    cos_az = np.clip(cos_az, -1.0, 1.0)
    az = np.degrees(np.arccos(cos_az))

    # Corrección de cuadrante según el signo del ángulo horario (mañana/tarde)
    az = np.where(hour_angle_rad > 0.0, (az + 180.0) % 360.0, (540.0 - az) % 360.0)
    return az.astype(np.float32)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray,
                      h_m, a: float, b: float):
    """
    Convierte coordenadas geodésicas (lat, lon, altura sobre el elipsoide)
    a ECEF, usando el MISMO elipsoide (a, b) que compute_latlon_grid lee de
    geometry.json (antes compute_local_zenith asumía una Tierra esférica
    y una longitud subsatelital hardcodeada, desacoplada de esto).
    """
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    e2 = 1.0 - (b ** 2) / (a ** 2)
    N = a / np.sqrt(1.0 - e2 * np.sin(lat_r) ** 2)

    x = (N + h_m) * np.cos(lat_r) * np.cos(lon_r)
    y = (N + h_m) * np.cos(lat_r) * np.sin(lon_r)
    z = (N * (1.0 - e2) + h_m) * np.sin(lat_r)
    return x, y, z


def compute_view_geometry(lat: np.ndarray, lon: np.ndarray,
                           sat_lon_deg: float, sat_height_m: float,
                           a: float, b: float):
    """
    Ángulo cenital local (LZA) y azimuth del satélite [deg], vistos desde
    cada píxel, calculados con geometría ECEF real. 
    Recibe lon_0/h/a/b directamente de
    geometry.json (vía load_geometry), el mismo archivo que usa
    compute_latlon_grid — así ambos cálculos quedan atados a la misma fuente.

    sat_height_m es la altura del satélite sobre el elipsoide en el punto
    subsatelital (perspective_point_height de geometry.json).
    """
    x_s, y_s, z_s = geodetic_to_ecef(0.0, sat_lon_deg, sat_height_m, a, b)
    x_g, y_g, z_g = geodetic_to_ecef(lat, lon, 0.0, a, b)

    dx, dy, dz = x_s - x_g, y_s - y_g, z_s - z_g
    rng = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)

    # Normal geodésica local (vertical local del elipsoide en el píxel)
    nx = np.cos(lat_r) * np.cos(lon_r)
    ny = np.cos(lat_r) * np.sin(lon_r)
    nz = np.sin(lat_r)

    cos_lza = (dx * nx + dy * ny + dz * nz) / rng
    cos_lza = np.clip(cos_lza, -1.0, 1.0)
    lza = np.degrees(np.arccos(cos_lza))

    # Vectores locales Este/Norte (ENU) para obtener el azimuth del satélite
    ex, ey, ez = -np.sin(lon_r), np.cos(lon_r), np.zeros_like(lon_r)
    nox = -np.sin(lat_r) * np.cos(lon_r)
    noy = -np.sin(lat_r) * np.sin(lon_r)
    noz =  np.cos(lat_r)

    e_comp = dx * ex + dy * ey + dz * ez
    n_comp = dx * nox + dy * noy + dz * noz
    sat_azimuth = np.degrees(np.arctan2(e_comp, n_comp)) % 360.0

    return lza.astype(np.float32), sat_azimuth.astype(np.float32)


def compute_local_zenith(lat: np.ndarray, lon: np.ndarray,
                          sat_lon_deg: float, sat_height_m: float,
                          a: float, b: float) -> np.ndarray:
    """
    Ángulo cenital local del satélite [deg] (wrapper de compute_view_geometry
    que devuelve solo el LZA, para no romper otros llamadores).
    Para GOES-19 (lon_0 ≈ -75°) sobre Uruguay da ~45-55°.
    """
    lza, _ = compute_view_geometry(lat, lon, sat_lon_deg, sat_height_m, a, b)
    return lza


def compute_glint_angle(sza: np.ndarray, lza: np.ndarray,
                         rel_azimuth: np.ndarray) -> np.ndarray:
    """
    Ángulo de glint solar [deg].
    Aproximación plana: ángulo entre el vector solar especular y el satélite.
    glint_angle ≈ 0 → glint perfecto. El FDCA usa threshold ~10°.

    rel_azimuth debe ser el azimuth relativo REAL (azimuth_satélite -
    azimuth_solar).
    """
    sza_r = np.radians(sza)
    lza_r = np.radians(lza)
    az_r  = np.radians(rel_azimuth)
    cos_glint = (np.cos(sza_r) * np.cos(lza_r) +
                 np.sin(sza_r) * np.sin(lza_r) * np.cos(az_r))
    cos_glint = np.clip(cos_glint, -1.0, 1.0)
    return np.degrees(np.arccos(cos_glint)).astype(np.float32)


# ── Conversiones radiométricas ────────────────────────────────────────────────

def rad_to_bt(band: int, rad: np.ndarray) -> np.ndarray:
    """
    Radiancia → Temperatura de Brillo usando Planck inverso.
    Reutiliza planck_temp del FDCA para consistencia exacta.
    """
    from fdca.planck import planck_temp
    bt = planck_temp(band, np.where(rad > 0, rad, np.nan))
    return bt.astype(np.float32)


def rad_b02_to_reflectance(rad: np.ndarray, sza: np.ndarray) -> np.ndarray:
    """
    Convierte radiancia B02 [W·m⁻²·sr⁻¹·µm⁻¹] a factor de reflectancia [-].

    refl = π * Rad / (ESUN_B02 * cos(SZA))

    Noche (SZA ≥ 90°) → NaN (no hay luz solar).
    El FDCA usa este valor como 'refl2' para los tests de nube y glint diurnos.
    """
    import math
    cos_sza = np.cos(np.radians(sza))
    # Evitar división por cero en píxeles nocturnos
    cos_sza_safe = np.where(cos_sza > 0.0, cos_sza, np.nan)
    refl = math.pi * rad / (ESUN_B02 * cos_sza_safe)
    return np.clip(refl, 0.0, 1.5).astype(np.float32)   # clip para sat/ruido


# ── Máscaras de superficie ────────────────────────────────────────────────────

def build_surface_masks(lat: np.ndarray, lon: np.ndarray,
                         region_name: str = "uruguay") -> dict:
    """
    Máscaras de superficie para la región.

    En producción ideal se usarían:
      - ABI ANC producto de land cover
      - MCD12Q1 (MODIS land cover)
      - Base de datos de desiertos brillantes del FDCA

    Aquí usamos una aproximación geográfica que es correcta para Uruguay/Cono Sur:
      - Todo el dominio de Uruguay es tierra (no hay grandes cuerpos de agua internos)
      - Sin desiertos brillantes (la región es pampa/campos)
      - USGS ecosystem: 10 = Grassland/savanna (representativo del Cono Sur)
      - MODIS land cover: 8 = Wooded grassland (válido para Uruguay)
    """
    H, W = lat.shape

    land_mask   = np.ones ((H, W), dtype=bool)
    land_cover  = np.full ((H, W), 8,  dtype=np.int32)   # Wooded grassland
    desert_mask = np.zeros((H, W),     dtype=np.int32)   # Sin desierto
    usgs_eco    = np.full ((H, W), 10, dtype=np.int32)   # Grassland/savanna

    # Para la región río_de_la_plata: el Río de la Plata es agua
    # Aproximación: lat < -34, lon entre -58 y -52 → zona del estuario
    if region_name in ("rio_de_la_plata",):
        water_zone = (lat < -34.0) & (lon > -58.0) & (lon < -52.0)
        land_mask  [water_zone] = False
        land_cover [water_zone] = 0    # agua
        usgs_eco   [water_zone] = 0

    return {
        "land_mask":   land_mask,
        "land_cover":  land_cover,
        "desert_mask": desert_mask,
        "usgs_eco":    usgs_eco,
    }


# ── LUT de corrección TPW ─────────────────────────────────────────────────────

def build_tpw_lut() -> np.ndarray:
    """
    LUT de corrección de vapor de agua precipitable (6 × 35).
    5 bins TPW × 7 bins ángulo zenith local.

    Filas:
      0: etiqueta bin TPW   (1–5)
      1: etiqueta bin ángulo (1–7)
      2: transmisión canal 7 (3.9 µm)
      3: transmisión canal 14 (11.2 µm)
      4: coeficiente absorción canal 7
      5: coeficiente absorción canal 14

    Valores basados en el ATBD de FDCA (valores representativos).
    Para producción real: usar el producto ABI-L2-TPWF para obtener TPW
    y consultar la tabla exacta del ATBD.
    """
    lut = np.zeros((6, 35))

    # Transmisión por bin: [TPW_bin 1..5] × [ang_bin 1..7]
    trans7 = np.array([
        [0.985, 0.983, 0.979, 0.973, 0.964, 0.950, 0.930],
        [0.972, 0.969, 0.963, 0.954, 0.941, 0.923, 0.898],
        [0.960, 0.956, 0.948, 0.937, 0.921, 0.899, 0.869],
        [0.949, 0.944, 0.935, 0.921, 0.903, 0.879, 0.845],
        [0.938, 0.932, 0.922, 0.907, 0.887, 0.861, 0.824],
    ])
    trans14 = np.array([
        [0.978, 0.975, 0.969, 0.960, 0.948, 0.931, 0.908],
        [0.957, 0.953, 0.946, 0.935, 0.920, 0.900, 0.874],
        [0.937, 0.933, 0.924, 0.911, 0.895, 0.873, 0.845],
        [0.918, 0.913, 0.903, 0.889, 0.872, 0.849, 0.819],
        [0.900, 0.895, 0.884, 0.869, 0.850, 0.826, 0.794],
    ])
    ext7  = trans7  * 0.005
    ext14 = trans14 * 0.008

    for tpw_bin in range(5):
        for ang_bin in range(7):
            col = tpw_bin * 7 + ang_bin
            lut[0, col] = tpw_bin + 1
            lut[1, col] = ang_bin + 1
            lut[2, col] = trans7 [tpw_bin, ang_bin]
            lut[3, col] = trans14[tpw_bin, ang_bin]
            lut[4, col] = ext7   [tpw_bin, ang_bin]
            lut[5, col] = ext14  [tpw_bin, ang_bin]
    return lut


def get_tpw_estimate(lat: np.ndarray, lon: np.ndarray,
                      dt: datetime) -> np.ndarray:
    """
    Estimación de TPW [mm] para la región.

    Opciones (en orden de precisión):
      A) Si tenés ABI-L2-TPWF descargado: leerlo directo (más preciso)
      B) Climatología mensual para la región (implementada acá)
      C) Constante 25 mm (fallback)

    La climatología mensual para Uruguay/Cono Sur (aproximada):
      Verano (dic-feb): ~35-45 mm
      Otoño  (mar-may): ~25-35 mm
      Invierno (jun-ago): ~15-25 mm
      Primavera (sep-nov): ~20-30 mm
    """
    month = dt.month
    if   month in (12, 1, 2):   base_tpw = 40.0
    elif month in (3, 4, 5):    base_tpw = 30.0
    elif month in (6, 7, 8):    base_tpw = 20.0
    else:                        base_tpw = 25.0

    # Gradiente latitudinal suave: más húmedo al norte
    lat_grad = (lat - lat.min()) / max(lat.max() - lat.min(), 1e-6)
    tpw = base_tpw + 10.0 * lat_grad   # +10 mm del sur al norte

    return tpw.astype(np.float32)


# ── Función principal ─────────────────────────────────────────────────────────

def load_fdca_input(
    timestamp: str,
    region:    str   = "uruguay",
    dataset_root: str = "dataset",
    config_path: str  = "config.yaml",
    verbose: bool = True,
) -> "FDCAInput":
    """
    Carga los .npy del pipeline para un timestamp dado y construye FDCAInput.

    Parameters
    ----------
    timestamp    : str   formato "YYYYMMDD_HHMM", ej. "20250905_1500"
    region       : str   nombre de región del config.yaml
    dataset_root : str   raíz del dataset (donde están las carpetas por banda)
    config_path  : str   ruta al config.yaml
    verbose      : bool  mostrar resumen de los arrays cargados

    Returns
    -------
    FDCAInput   (importado del módulo fdca)

    Raises
    ------
    FileNotFoundError  si faltan B07 o B14 (inputs mínimos obligatorios)
    """
    import yaml
    #sys.path.insert(0, str(Path(config_path).parent.parent))
    from .algorithm import FDCAInput

    # ── Cargar config ──────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    region_cfg = cfg["regions"][region]
    base = os.path.join(dataset_root, region)

    # ── Parsear timestamp ──────────────────────────────────────────────────────
    #dt = datetime.strptime(timestamp, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc) antes
    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M")  # sin timezone, fdca.py lo maneja internamente

    # ── Función auxiliar para leer .npy ────────────────────────────────────────
    def load_band(band_id: str, required: bool = False) -> np.ndarray | None:
        path = os.path.join(base, band_id, f"{timestamp}.npy")
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(
                    f"Input obligatorio no encontrado: {path}\n"
                    f"Corré: python pipeline.py download --region {region} "
                    f"--start '{dt.strftime('%Y-%m-%d %H:%M')}' "
                    f"--end '{dt.strftime('%Y-%m-%d %H:%M')}' "
                    f"--products {band_id}"
                )
            return None
        return np.load(path)
    
   
    def load_planck_coeffs(base: str, band_id: str, timestamp: str) -> dict | None:
        """Lee los coeficientes Planck (*_planck.json) generados por downloader.py
        y los mapea a los nombres esperados por planck_temp_from_coeffs (fk1, fk2, bc1, bc2)."""
        path = os.path.join(base, band_id, f"{timestamp}_planck.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            raw = json.load(f)
        return {
            "fk1": raw["planck_fk1"],
            "fk2": raw["planck_fk2"],
            "bc1": raw["planck_bc1"],
            "bc2": raw["planck_bc2"],
        }

    


    from fdca.planck import planck_temp_from_coeffs, planck_rad

    ## Calculos de lo necesario para correr el algoritmo

    #Mientras que las bandas térmicas miden el calor emitido por la Tierra (en número de onda), 
    #la banda B02 mide la luz del Sol reflejada por la superficie y la atmósfera (en longitud de onda).

    # Radiancias crudas tal cual vienen del .nc — SIN conversión manual de unidades.
    # Su unidad nativa es mW m-2 sr-1 (cm-1)-1, y se invierten a BT usando
    # los coeficientes Planck propios de cada archivo (planck_temp_from_coeffs).
    rad7_raw  = load_band("ABI-L1b-Rad-B07", required=True)
    rad14_raw = load_band("ABI-L1b-Rad-B14", required=True)
    rad13_raw = load_band("ABI-L1b-Rad-B13")
    rad15_raw = load_band("ABI-L1b-Rad-B15")

    # Cargar coeficientes Planck para B07/B14 (obligatorios) 
    coeffs7  = load_planck_coeffs(base, "ABI-L1b-Rad-B07", timestamp)
    coeffs14 = load_planck_coeffs(base, "ABI-L1b-Rad-B14", timestamp)

    if coeffs7 is None or coeffs14 is None:
            raise FileNotFoundError(
                f"Faltan coeficientes Planck (*_planck.json) para {timestamp}.\n"
                f"Re-descargá B07/B14 con la versión actualizada de downloader.py "
                f"(borrá los .npy existentes y volvé a correr 'pipeline.py download')."
            )
    

    #radiancia cruda del .nc -> temperatura de brillo [K].
    bt7  = planck_temp_from_coeffs(rad7_raw,  **coeffs7).astype(np.float32)
    bt14 = planck_temp_from_coeffs(rad14_raw, **coeffs14).astype(np.float32)


    coeffs13 = load_planck_coeffs(base, "ABI-L1b-Rad-B13", timestamp)
    bt13 = (planck_temp_from_coeffs(rad13_raw, **coeffs13).astype(np.float32)
            if (rad13_raw is not None and coeffs13 is not None) else None)

    coeffs15 = load_planck_coeffs(base, "ABI-L1b-Rad-B15", timestamp)
    bt15 = (planck_temp_from_coeffs(rad15_raw, **coeffs15).astype(np.float32)
            if (rad15_raw is not None and coeffs15 is not None) else None)

    
    # Radiancias crudas tal cual vienen del .nc
    # Su unidad nativa es W * m^-2 * sr^-1 * µm^-1
    rad02     = load_band("ABI-L1b-Rad-B02", required=False)
    

    # ── Regenerar radiancias "consistentes" en W·m⁻²·sr⁻¹·m⁻¹ ────────────────
    # El resto del pipeline (Dozier, background, FRP) fue validado con
    # planck_rad/planck_temp en esta unidad. Para mantener consistencia
    # interna, recalculamos rad7/rad14/rad13/rad15 a partir del BT correcto
    # (ya invertido con los coeficientes reales) usando planck_rad genérico.
    # TODO: reescribir Dozier/background/FRP para trabajar nativamente en
    # mW m-2 sr-1 (cm-1)-1 y eliminar este paso (ver fix riguroso pendiente).
    rad7  = planck_rad(7,  bt7)
    rad14 = planck_rad(14, bt14)
    rad13 = planck_rad(13, bt13) 
    rad15 = planck_rad(15, bt15) 

    if verbose:
        def band_info(name, arr):
            if arr is None:
                return f"  {name:<22}: ✗ no disponible"
            return f"  {name:<22}: shape={arr.shape}  range=[{np.nanmin(arr):.2f}, {np.nanmax(arr):.2f}]"
        print(band_info("ABI-L1b-Rad-B07", rad7))
        print(band_info("ABI-L1b-Rad-B14", rad14))
        print(band_info("ABI-L1b-Rad-B13", rad13))
        print(band_info("ABI-L1b-Rad-B15", rad15))
        print(band_info("ABI-L1b-Rad-B02", rad02))

    shape = rad7.shape

    # ── Grilla lat/lon ────────────────────────────────────────────────────────
    lat2d, lon2d = compute_latlon_grid(base)

    # ── Geometría del satélite (misma fuente que usó compute_latlon_grid) ────
    geom = load_geometry(base)

    # ── Geometría solar y satelital ───────────────────────────────────────────
    sza          = compute_solar_zenith(lat2d, lon2d, dt)
    solar_az     = compute_solar_azimuth(lat2d, lon2d, dt)
    lza, sat_az  = compute_view_geometry(
        lat2d, lon2d,
        sat_lon_deg=geom["lon_0"], sat_height_m=geom["h"],
        a=geom["a"], b=geom["b"],
    )
    # Azimuth relativo real sol-satélite (antes: constante 120° para toda la imagen)
    azimuth = (sat_az - solar_az) % 360.0
    glint   = compute_glint_angle(sza, lza, azimuth)

    if verbose:
        print(f"\n  {'SZA range [°]':<22}: {sza.min():.1f} – {sza.max():.1f}")
        print(f"  {'LZA range [°]':<22}: {lza.min():.1f} – {lza.max():.1f}")
        day_pct = 100 * (sza <= 85).mean()
        print(f"  {'Píxeles diurnos':<22}: {day_pct:.0f}%")


    # ── Reflectancia B02 ──────────────────────────────────────────────────────
    if rad02 is not None:
        # Submuestrear B02 de (893,1212) a (224,303) promediando bloques
        from PIL import Image
        H, W = shape
        img = Image.fromarray(rad02).resize((W, H), Image.BILINEAR)
        rad02_resized = np.array(img, dtype=np.float32)
        refl2 = rad_b02_to_reflectance(rad02_resized, sza)
    else:
        refl2 = None

    if verbose:
        print(f"\n  {'BT7 range [K]':<22}: {np.nanmin(bt7):.1f} – {np.nanmax(bt7):.1f}")
        print(f"  {'BT14 range [K]':<22}: {np.nanmin(bt14):.1f} – {np.nanmax(bt14):.1f}")
        if refl2 is not None:
            print(f"  {'refl2 range':<22}: {np.nanmin(refl2):.3f} – {np.nanmax(refl2):.3f}")


    # ── Emissividad ────────────────────────────────────────────────────────────
    # Valores típicos de vegetación/suelo para Uruguay
    # Para producción: usar ABI ANC emissivity product (si está disponible)
    emiss7  = np.full(shape, 0.95, dtype=np.float32)
    emiss14 = np.full(shape, 0.97, dtype=np.float32)

    # ── TPW ───────────────────────────────────────────────────────────────────
    tpw = get_tpw_estimate(lat2d, lon2d, dt)
    if verbose:
        print(f"  {'TPW range [mm]':<22}: {tpw.min():.1f} – {tpw.max():.1f}  (estimación climatológica)")

    # ── Máscaras de superficie ────────────────────────────────────────────────
    masks = build_surface_masks(lat2d, lon2d, region_name=region)

    # ── LUT TPW ───────────────────────────────────────────────────────────────
    lut_tpw = build_tpw_lut()

    # ── FPT: Focal Plane Temperature de ABI ──────────────────────────────────
    # ABI en GOES-19 opera a ~85-87 K (criogénico) → por debajo del umbral 90 K
    # → no se activa el modo híbrido de B13 (FPT_THRESHOLD = 90 K en constants.py)
    FPT = 85.0

    # ── Armar FDCAInput ───────────────────────────────────────────────────────
    inp = FDCAInput(
        bt7=bt7,   rad7=rad7,
        bt14=bt14, rad14=rad14,
        bt13=bt13, rad13=rad13,
        bt15=bt15, rad15=rad15,
        refl2=refl2,
        latitudes=lat2d, longitudes=lon2d,
        sza=sza, glint_angle=glint,
        lza=lza, azimuth=azimuth,
        tpw=tpw, emiss7=emiss7, emiss14=emiss14,
        lut_tpw=lut_tpw, FPT=FPT,
        land_cover=masks["land_cover"],
        land_mask=masks["land_mask"],
        desert_mask=masks["desert_mask"],
        usgs_eco=masks["usgs_eco"],
        scan_time=dt,
        prev_fire_mask=None,
        data_quality=None,
    )

    if verbose:
        print(f"\n  ✓ FDCAInput construido — shape {shape}")

    return inp