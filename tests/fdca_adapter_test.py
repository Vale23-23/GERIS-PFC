"""
test_fdca_adapter.py
────────────────────
Suite de pruebas unitarias e integración para la capa de adaptación de datos.
Garantiza que la traducción de formatos, máscaras y geometría sea consistente.
"""

import os
import json
import yaml
import pytest
import numpy as np
from datetime import datetime

# Importar todas las funciones del adaptador bajo prueba
import fdca_adapter as adapter

# ── FIXTURES PARA MOCKEAR DATOS DE ENTRADA ────────────────────────────────────

@pytest.fixture
def mock_dataset_root(tmp_path):
    """
    Crea un entorno mínimo de archivos simulando la estructura del dataset recortado
    generado por el downloader para la región de Uruguay.
    """
    region = "uruguay"
    timestamp = "20260706_1500"
    
    # Rutas de carpetas simuladas
    base_dir = tmp_path / region
    b07_dir = base_dir / "ABI-L1b-Rad-B07"
    b14_dir = base_dir / "ABI-L1b-Rad-B14"
    
    b07_dir.mkdir(parents=True, exist_ok=True)
    b14_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Crear geometry.json simulado (4x4 píxeles en el espacio de Uruguay)
    # Valores de radianes de escaneo fijos del GOES-19
    geom_data = {
        "x": [-0.0152, -0.0151, -0.0150, -0.0149],
        "y": [0.0865, 0.0864, 0.0863, 0.0862],
        "longitude_of_projection_origin": -75.0,
        "perspective_point_height": 35786023.0,
        "semi_major_axis": 6378137.0,
        "semi_minor_axis": 6356752.3
    }
    with open(b07_dir / "geometry.json", "w") as f:
        json.dump(geom_data, f)
        
    # 2. Crear coeficientes de Planck mockeados para B07 y B14
    planck_b07 = {"planck_fk1": 6023.0, "planck_fk2": 3678.0, "planck_bc1": 0.43, "planck_bc2": 0.99}
    planck_b14 = {"planck_fk1": 2026.0, "planck_fk2": 1282.0, "planck_bc1": 0.22, "planck_bc2": 0.98}
    
    with open(b07_dir / f"{timestamp}_planck.json", "w") as f:
        json.dump(planck_b07, f)
    with open(b14_dir / f"{timestamp}_planck.json", "w") as f:
        json.dump(planck_b14, f)
        
    # 3. Crear matrices .npy crudas de radianzas (4x4)
    np.save(b07_dir / f"{timestamp}.npy", np.full((4, 4), 1.5, dtype=np.float32))
    np.save(b14_dir / f"{timestamp}.npy", np.full((4, 4), 75.0, dtype=np.float32))
    
    # 4. Crear archivo config.yaml mínimo en la raíz temporal
    config_data = {
        "regions": {
            "uruguay": {"lat_min": -35.0, "lat_max": -30.0, "lon_min": -58.5, "lon_max": -53.0}
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    return {
        "dataset_root": str(tmp_path),
        "config_path": str(config_file),
        "timestamp": timestamp,
        "region": region,
        "b07_path": str(b07_dir)
    }


# ── 1. TESTS DE GEOMETRÍA Y PROYECCIÓN ABI ────────────────────────────────────

def test_compute_latlon_grid(mock_dataset_root):
    base_b07 = mock_dataset_root["b07_path"]
    lat, lon = adapter.compute_latlon_grid(base_b07)
    
    # Verificar dimensiones físicas de la grilla
    assert lat.shape == (4, 4)
    assert lon.shape == (4, 4)
    assert lat.dtype == np.float32
    
    # Verificar límites geográficos esperados para el Cono Sur (Uruguay)
    assert np.nanmin(lat) < -30.0
    assert np.nanmax(lat) > -31.5
    assert np.nanmin(lon) < -53.0
    assert np.nanmax(lon) > -57.0


# ── 2. TESTS DE POSICIÓN SOLAR (SZA / AZIMUTH) ────────────────────────────────

def test_compute_solar_zenith_day_and_night():
    # Coordenadas estáticas sobre Uruguay
    lat = np.array([[-34.5]])
    lon = np.array([[-56.0]])
    
    # Test Mediodía (SZA debe ser bajo)
    dt_noon = datetime(2026, 1, 1, 15, 0) # 15:00 UTC ≈ 12:00 local
    sza_noon = adapter.compute_solar_zenith(lat, lon, dt_noon)
    assert 0.0 <= sza_noon[0, 0] <= 90.0
    assert sza_noon[0, 0] < 35.0 # En enero el sol está alto en Uruguay
    
    # Test Noche profunda (SZA debe superar los 90°)
    dt_night = datetime(2026, 1, 1, 3, 0) # 03:00 UTC ≈ 00:00 local
    sza_night = adapter.compute_solar_zenith(lat, lon, dt_night)
    assert sza_night[0, 0] > 90.0


# ── 3. TESTS DE GEOMETRÍA DE VISIÓN DEL SATÉLITE (LZA / AZIMUTH) ──────────────

def test_compute_view_geometry():
    # Uruguay central
    lat = np.array([[-33.0]], dtype=np.float32)
    lon = np.array([[-56.0]], dtype=np.float32)
    
    # Parámetros del GOES-19
    lza, sat_az = adapter.compute_view_geometry(
        lat, lon, sat_lon_deg=-75.0, sat_height_m=35786023.0, 
        a=6378137.0, b=6356752.3
    )
    
    # Un satélite a -75°W mirando a Uruguay siempre caerá en estos rangos
    assert 40.0 < lza[0, 0] < 60.0
    assert 300.0 < sat_az[0, 0] < 350.0 # Cuadrante Noroeste


# ── 4. TESTS DEL ÁNGULO DE REFLEXIÓN ESPECULAR (GLINT) ────────────────────────

def test_compute_glint_angle_limits():
    # Caso Espejo Perfecto (Alineación total vector sol y satélite)
    sza = np.array([[30.0]])
    lza = np.array([[30.0]])
    rel_az_perfect = np.array([[180.0]]) # Especular opuesto
    
    glint_perfect = adapter.compute_glint_angle(sza, lza, rel_az_perfect)
    assert np.isclose(glint_perfect[0, 0], 0.0, atol=1e-2)
    
    # Caso No-Glint (Vectores ortogonales)
    rel_az_far = np.array([[0.0]])
    glint_far = adapter.compute_glint_angle(sza, lza, rel_az_far)
    assert glint_far[0, 0] > 50.0


# ── 5. TESTS DE CONVERSIÓN DE REFLECTANCIA (BANDA VISIBLE B02) ────────────────

def test_rad_b02_to_reflectance():
    rad = np.full((2, 2), 150.0, dtype=np.float32)
    
    # Escenario de Día (SZA = 0, Máxima insolación)
    sza_day = np.zeros((2, 2), dtype=np.float32)
    refl_day = adapter.rad_b02_to_reflectance(rad, sza_day)
    assert np.all(refl_day > 0.0)
    assert np.all(refl_day <= 1.5)
    
    # Escenario de Noche (SZA = 95°, Sin sol directo -> Debe dar NaN)
    sza_night = np.full((2, 2), 95.0, dtype=np.float32)
    refl_night = adapter.rad_b02_to_reflectance(rad, sza_night)
    assert np.all(np.isnan(refl_night))


# ── 6. TESTS DE MÁSCARAS DE SUPERFICIE ────────────────────────────────────────

def test_build_surface_masks():
    # Crear una pequeña matriz regional sintética
    lat = np.array([[-31.0, -31.0], [-34.5, -34.5]])
    lon = np.array([[-55.0, -53.0], [-55.0, -53.0]])
    
    # Región Continental: Uruguay regular
    masks_land = adapter.build_surface_masks(lat, lon, region_name="uruguay")
    assert masks_land["land_mask"].all() # Toda la pampa es tierra firme
    assert np.all(masks_land["land_cover"] == 8)
    
    # Región Estuario: Río de la Plata (Debe detectar agua en la esquina inferior)
    masks_water = adapter.build_surface_masks(lat, lon, region_name="rio_de_la_plata")
    assert not masks_water["land_mask"].all() # Tiene que haber píxeles de agua
    assert masks_water["land_mask"][1, 1] == False # Coordenada aproximada del estuario


# ── 7. TESTS DE ESTIMACIÓN DE AGUA PRECIPITABLE (TPW) Y LUT ───────────────────

def test_get_tpw_estimate_seasonal():
    lat = np.array([[-33.0]])
    lon = np.array([[-56.0]])
    
    tpw_jan = adapter.get_tpw_estimate(lat, lon, datetime(2026, 1, 15))
    tpw_jul = adapter.get_tpw_estimate(lat, lon, datetime(2026, 7, 15))
    
    # El verano del hemisferio sur es sustancialmente más húmedo que el invierno
    assert tpw_jan.mean() > tpw_jul.mean()


def test_build_tpw_lut():
    lut = adapter.build_tpw_lut()
    
    assert lut.shape == (6, 35)
    # Transmisividades (filas indexadas 2 y 3) de la física atmosférica siempre acotadas entre 0 y 1
    assert np.all(lut[2] >= 0.0) and np.all(lut[2] <= 1.0)
    assert np.all(lut[3] >= 0.0) and np.all(lut[3] <= 1.0)


# ── 8. TEST DE INTEGRACIÓN: load_fdca_input() ─────────────────────────────────

def test_load_fdca_input_integration(mock_dataset_root):
    """
    Test de integración definitivo de la capa de datos. Valida que el pipeline
    construya un objeto FDCAInput balanceado, alineado espacialmente y sin NaNs
    críticos en los vectores obligatorios del core matemático.
    """
    inp = adapter.load_fdca_input(
        timestamp=mock_dataset_root["timestamp"],
        region=mock_dataset_root["region"],
        dataset_root=mock_dataset_root["dataset_root"],
        config_path=mock_dataset_root["config_path"],
        verbose=False
    )
    
    # Verificar consistencia de la estructura de datos empaquetada
    assert inp.bt7.shape == (4, 4)
    assert inp.bt14.shape == (4, 4)
    assert inp.latitudes.shape == (4, 4)
    assert inp.sza.shape == (4, 4)
    assert inp.lza.shape == (4, 4)
    assert inp.tpw.shape == (4, 4)
    
    # Verificar flags fijos globales
    assert inp.FPT == 85.0
    assert inp.scan_time.year == 2026
    
    # Regla de Oro: No puede haber NaNs en las matrices de control de ángulos obligatorios
    assert not np.isnan(inp.sza).any()
    assert not np.isnan(inp.lza).any()
    assert not np.isnan(inp.azimuth).any()