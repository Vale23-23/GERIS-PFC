import json
import os
import numpy as np

MANIFEST_FILE = "manifest.json"

def load(output_root):
    path = os.path.join(output_root, MANIFEST_FILE)
    if not os.path.exists(path): 
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return {}
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
    Updates the manifest with bands and fire statistics.
    """
    manifest = load(output_root)

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

        if prod_id == "ABI-L2-FDCF" and res["status"] in ("downloaded", "exists"):
            file_path = res.get("path")
            if file_path and os.path.exists(file_path):
                mask = np.load(file_path)
                
                # In this pipeline, ABI-L2-FDCF is already encoded as 0 = high-confidence fire.
                fire_pixels = int(np.sum(mask == 0))
                has_fire = fire_pixels > 0
                        
                fire_meta = {
                    "fire_pixels": fire_pixels,
                    "has_fire": has_fire,
                    "class_label": "fire" if has_fire else "clear"
                }
                
                manifest[ts]["fire"] = fire_meta

        manifest[ts]["bands"][prod_id] = band_data

    # Recalculate timestamp integrity
    for ts, data in manifest.items():
        ok_products = {p for p, info in data["bands"].items() if info["status"] in ("downloaded", "exists")}
        data["status"] = "complete" if all(pid in ok_products for pid in all_required_ids) else "incomplete"

    save(output_root, manifest)
    return manifest

def export_metadata_csv(output_root):
    """Flattens the JSON to a complete CSV with all available information."""
    import csv
    data = load(output_root)
    if not data:
        return

    csv_path = os.path.join(output_root, "metadata.csv")
    fieldnames = [
        'timestamp', 'status', 'has_fire', 'class_label', 'total_fire_pixels',
        'ABI-L1b-Rad-B07_status', 'ABI-L1b-Rad-B07_shape', 'ABI-L1b-Rad-B07_path',
        'ABI-L1b-Rad-B14_status', 'ABI-L1b-Rad-B14_shape', 'ABI-L1b-Rad-B14_path',
        'ABI-L2-FDCF_status', 'ABI-L2-FDCF_shape', 'ABI-L2-FDCF_path'
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ts, info in sorted(data.items()):
            fire = info.get("fire", {})
            row = {
                'timestamp': ts,
                'status': info.get('status', 'incomplete'),
                'has_fire': fire.get("has_fire", False),
                'class_label': fire.get("class_label", "unknown"),
                'total_fire_pixels': fire.get("fire_pixels", 0)
            }

            # Add information to each band
            bands = info.get("bands", {})
            for band_id in ['ABI-L1b-Rad-B07', 'ABI-L1b-Rad-B14', 'ABI-L2-FDCF']:
                band_info = bands.get(band_id, {})
                row[f'{band_id}_status'] = band_info.get("status", "missing")
                row[f'{band_id}_shape'] = str(band_info.get("shape", "N/A"))
                row[f'{band_id}_path'] = band_info.get("path", "N/A")

            writer.writerow(row)

def status_report(output_root, required_products=None):
    manifest = load(output_root)
    if not manifest:
        print("Manifest is empty.")
        return
    total = len(manifest)
    complete = sum(1 for v in manifest.values() if v.get("status") == "complete")
    print(f"\n📊 Status at {output_root}: {complete}/{total} complete.")