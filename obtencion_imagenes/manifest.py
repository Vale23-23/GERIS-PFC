"""
manifest.py — tracks downloaded files and dataset integrity.

The manifest is a JSON file at {output_root}/manifest.json.
Schema per entry:
  {
    "timestamp": "20250901_1100",
    "product":   "ABI-L1b-Rad-B07",
    "status":    "downloaded" | "error" | "empty" | "exists",
    "path":      "dataset/ABI-L1b-Rad-B07/20250901_1100.npy",
    "shape":     [H, W],          # only when status == downloaded/exists
    "error":     "...",           # only when status == error
  }
"""

import json
import os
from datetime import datetime


MANIFEST_FILE = "manifest.json"


def _manifest_path(output_root):
    return os.path.join(output_root, MANIFEST_FILE)


def load(output_root):
    path = _manifest_path(output_root)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save(output_root, entries):
    path = _manifest_path(output_root)
    os.makedirs(output_root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def update(output_root, new_results):
    """Merge new download results into the manifest (upsert by timestamp+product)."""
    entries = load(output_root)

    # Build lookup for fast upsert
    index = {(e["timestamp"], e["product"]): i for i, e in enumerate(entries)}

    for result in new_results:
        key = (result["timestamp"], result["product"])
        if key in index:
            entries[index[key]] = result
        else:
            entries.append(result)
            index[key] = len(entries) - 1

    save(output_root, entries)
    return entries


def status_report(output_root, required_products=None):
    """
    Print a summary of the manifest.
    If required_products is given, also reports timestamps with incomplete coverage.
    """
    entries = load(output_root)
    if not entries:
        print("Manifest vacío o no encontrado.")
        return

    from collections import defaultdict
    by_product = defaultdict(lambda: {"downloaded": 0, "error": 0, "empty": 0, "exists": 0})
    by_ts      = defaultdict(set)

    for e in entries:
        st = e.get("status", "error")
        by_product[e["product"]][st] += 1
        if st in ("downloaded", "exists"):
            by_ts[e["timestamp"]].add(e["product"])

    print("\n📦 Estado por producto:")
    for prod, counts in sorted(by_product.items()):
        ok    = counts["downloaded"] + counts["exists"]
        fails = counts["error"] + counts["empty"]
        print(f"  {prod:<30} ✅ {ok}  ❌ {fails}")

    if required_products:
        req = set(required_products)
        complete   = [ts for ts, prods in by_ts.items() if req.issubset(prods)]
        incomplete = [ts for ts, prods in by_ts.items() if not req.issubset(prods)]
        print(f"\n✅ Timestamps completos (todos los productos): {len(complete)}")
        if incomplete:
            print(f"⚠️  Timestamps incompletos: {len(incomplete)}")
            for ts in sorted(incomplete)[:10]:
                missing = req - by_ts[ts]
                print(f"    {ts}  falta: {', '.join(missing)}")
