import json
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

MANIFEST_FILE = "manifest.json"
GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "departamentos.geojson")

def load(output_root):
    path = os.path.join(output_root, MANIFEST_FILE)
    if not os.path.exists(path): 
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return {} # Reset si detecta formato viejo
            return data
    except:
        return {}

def save(output_root, entries):
    path = os.path.join(output_root, MANIFEST_FILE)
    os.makedirs(output_root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

def update(output_root, new_results, all_required_ids, region_cfg):
    """
    Actualiza el manifest con bandas, estadísticas de fuego y reporte por departamento.
    """
    manifest = load(output_root)
    
    # Intentamos cargar el mapa oficial una sola vez para esta tanda
    gdf_deptos = None
    if os.path.exists(GEOJSON_PATH):
        try:
            gdf_deptos = gpd.read_file(GEOJSON_PATH)
        except Exception as e:
            print(f"⚠️ Error cargando GeoJSON: {e}")

    for res in new_results:
        ts = res["timestamp"]
        prod_id = res["product"]
        
        if ts not in manifest:
            manifest[ts] = {"status": "incomplete", "bands": {}}

        shape = res.get("shape")
        if shape is None:
            shape = manifest[ts]["bands"].get(prod_id, {}).get("shape")

        band_data = {
            "status": res["status"],
            "path": res.get("path"), 
            "shape": shape,
            "error": res.get("error")
        }

        # --- Lógica de Fuego y Espacial ---
        if prod_id == "ABI-L2-FDCF" and res["status"] in ("downloaded", "exists"):
            file_path = res.get("path")
            if file_path and os.path.exists(file_path):
                mask = np.load(file_path)
                
                # En este pipeline, ABI-L2-FDCF ya viene codificado como 0 = fuego de alta confianza.
                # Por eso aquí debemos buscar el valor 0, no los códigos GOES-R originales.
                fire_pixels = int(np.sum(mask == 0))
                has_fire = fire_pixels > 0
                        
                fire_meta = {
                    "fire_pixels": fire_pixels,
                    "has_fire": has_fire,
                    "class_label": "fire" if has_fire else "clear"
                }
                
                # Si hay fuego, calculamos en qué departamentos cayó
                if has_fire and gdf_deptos is not None:
                    rows, cols = np.where(mask == 0)
                    h, w = mask.shape
                    
                    # Interpolación de coordenadas según config.yaml
                    lats = region_cfg['lat_max'] - (rows / h) * (region_cfg['lat_max'] - region_cfg['lat_min'])
                    lons = region_cfg['lon_min'] + (cols / w) * (region_cfg['lon_max'] - region_cfg['lon_min'])
                    
                    points = [Point(xy) for xy in zip(lons, lats)]
                    gdf_fire = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
                    
                    # Unión espacial con los departamentos
                    joined = gpd.sjoin(gdf_fire, gdf_deptos, predicate='within')
                    
                    # Usamos 'admlnm' que es la clave en tu archivo departamentos.geojson
                    if not joined.empty:
                        spatial_report = joined['admlnm'].value_counts().to_dict()
                        fire_meta["spatial_report"] = spatial_report
                
                manifest[ts]["fire"] = fire_meta

        manifest[ts]["bands"][prod_id] = band_data

    # Recalcular integridad del timestamp
    for ts, data in manifest.items():
        ok_products = {p for p, info in data["bands"].items() if info["status"] in ("downloaded", "exists")}
        data["status"] = "complete" if all(pid in ok_products for pid in all_required_ids) else "incomplete"

    save(output_root, manifest)
    export_metadata_csv(output_root) # Genera automáticamente el CSV para Hugging Face
    return manifest

def export_metadata_csv(output_root):
    """Aplatana el JSON a un CSV de 19 columnas (departamentos) para Hugging Face."""
    import csv
    data = load(output_root)
    if not data: 
        return

    # Lista exacta de tu GeoJSON
    deptos = [
        'Artigas', 'Canelones', 'Cerro Largo', 'Colonia', 'Durazno', 'Flores', 
        'Florida', 'Lavalleja', 'Maldonado', 'Montevideo', 'Paysandú', 'Rivera', 
        'Rocha', 'Río Negro', 'Salto', 'San José', 'Soriano', 'Tacuarembó', 'Treinta y tres'
    ]

    csv_path = os.path.join(output_root, "metadata.csv")
    fieldnames = ['timestamp', 'status', 'total_fire_pixels'] + deptos

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ts, info in sorted(data.items()):
            fire = info.get("fire", {})
            spatial = fire.get("spatial_report", {})
            row = {
                'timestamp': ts,
                'status': info.get('status', 'incomplete'),
                'total_fire_pixels': fire.get("fire_pixels", 0)
            }
            for d in deptos:
                row[d] = spatial.get(d, 0)
            writer.writerow(row)

def status_report(output_root, required_products=None):
    manifest = load(output_root)
    if not manifest:
        print("Manifest vacío.")
        return
    total = len(manifest)
    complete = sum(1 for v in manifest.values() if v.get("status") == "complete")
    print(f"\n📊 Estado en {output_root}: {complete}/{total} completos.")