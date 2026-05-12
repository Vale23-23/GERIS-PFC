"""
static_data_downloader.py — Download and prepare static ancillary data.

Downloads once, crops to region bounding box, saves as .npy to data/static/.
Never re-downloads if files already exist.

Files produced (all float32 or int8, shape matches ABI grid for the region):
  data/static/
    emissivity_b07.npy     — Surface emissivity for ABI band 7  (3.9 µm)
    emissivity_b14.npy     — Surface emissivity for ABI band 14 (11.2 µm)
    land_sea_mask.npy      — 1=land, 0=sea  (int8)
    land_cover.npy         — GlobCover land cover class (int16)

Sources (all public, no authentication required):
  Emissivities : NASA GSFC MODIS UMD global emissivity monthly climatology
                 https://neo.gsfc.nasa.gov/  (NetCDF, open HTTP)
  Land/sea mask: GSHHG via NOAA/NGDC — rasterised with rasterio + shapely
  Land cover   : ESA GlobCover 2009   — open HTTP from ESA Ionia server

Usage:
    python static_data_downloader.py              # uses config.yaml
    python static_data_downloader.py --region uruguay --config config.yaml

Dependencies (add to requirements.txt):
    netCDF4
    rasterio
    shapely
    fiona
    requests
"""

import argparse
import os
import tempfile
import zipfile

import numpy as np
import requests
import yaml

# ── Output folder ─────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join("data", "static")

# ── Remote URLs (no auth required) ────────────────────────────────────────────

# NASA NEO monthly emissivity climatology (Band 31 ≈ 11 µm, Band 20 ≈ 3.9 µm)
# These are global NetCDF files (~20 MB each).
_EMIS_B07_URL = (
    "https://neo.gsfc.nasa.gov/servlet/RenderData"
    "?imageid=MODAL2_M_SKY_WV&cs=gs&format=NetCDF"
    # We use the closest MODIS bands to ABI 7 (B20, ~3.7-3.9 µm)
    # Full URL resolved at runtime — see _download_emissivity()
)

# Better source: the CIMSS UW global emissivity database, mirrored on a public
# NASA GSFC FTP (anonymous access).
_EMIS_GLOBAL_URL = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/1000/1371/"
    "global_emis_inf10_monthFilled_MYD11C3_M_2010-2016.nc"
)

# GSHHG full resolution (land polygons) — from SOEST/UHawaii mirror
_GSHHG_URL = "https://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip"

# ESA GlobCover 2009 — direct download from ESA Ionia (no auth)
_GLOBCOVER_URL = (
    "http://due.esrin.esa.int/files/Globcover2009_V2.3_Global_.zip"
)

_TIMEOUT = 120  # seconds


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download(url: str, dest: str, label: str):
    """Stream-download url to dest, showing a simple progress indicator."""
    print(f"  ⬇  {label} ...", end=" ", flush=True)
    resp = requests.get(url, timeout=_TIMEOUT, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    written = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(1 << 17):  # 128 KB chunks
            f.write(chunk)
            written += len(chunk)
    mb = written / 1e6
    print(f"OK ({mb:.1f} MB)")


def _linspace_grid(region_cfg: dict, shape: tuple[int, int]):
    """
    Return (lats_1d, lons_1d) for a regular grid that matches an ABI .npy crop.
    Row 0 = north (lat_max), last row = south (lat_min).
    """
    rows, cols = shape
    lats = np.linspace(region_cfg["lat_max"], region_cfg["lat_min"], rows)
    lons = np.linspace(region_cfg["lon_min"], region_cfg["lon_max"], cols)
    return lats, lons


def _infer_goes_shape(output_root: str) -> tuple[int, int] | None:
    """Find any saved ABI .npy and return its shape."""
    for product in ("ABI-L1b-Rad-B07", "ABI-L1b-Rad-B14", "ABI-L1b-Rad-B13"):
        folder = os.path.join(output_root, product)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(folder, fname), mmap_mode="r")
                return arr.shape
    return None


# ── Emissivities ──────────────────────────────────────────────────────────────

def _build_emissivity(region_cfg: dict, shape: tuple[int, int], static_dir: str):
    """
    Download the NASA GSFC global emissivity NetCDF and extract bands 7 and 14.

    The file contains monthly climatology fields named:
      'emis_b07'  (3.9 µm, closest to ABI band 7)
      'emis_b14'  (11.2 µm, closest to ABI band 14)
    We take the annual mean across months and regrid to the ABI grid.
    """
    import netCDF4 as nc
    from scipy.interpolate import RegularGridInterpolator

    out_b07 = os.path.join(static_dir, "emissivity_b07.npy")
    out_b14 = os.path.join(static_dir, "emissivity_b14.npy")
    if os.path.exists(out_b07) and os.path.exists(out_b14):
        print("  ✅ Emissivities already exist, skipping.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        nc_path = os.path.join(tmp, "emis_global.nc")
        _download(_EMIS_GLOBAL_URL, nc_path, "Global emissivity NetCDF")

        ds = nc.Dataset(nc_path)

        # Coordinates — the file uses 0.05° global grid
        lats_global = ds.variables["lat"][:]   # shape (3600,) descending
        lons_global = ds.variables["lon"][:]   # shape (7200,) ascending

        # The file has separate variables for each MODIS band.
        # Band 20 (~3.7 µm) → closest to ABI band 7
        # Band 31 (~11.0 µm) → closest to ABI band 14
        # Variables are named 'emis20' and 'emis31', monthly (12, lat, lon).
        emis_b07_monthly = ds.variables["emis20"][:]   # (12, lat, lon)
        emis_b14_monthly = ds.variables["emis31"][:]   # (12, lat, lon)
        ds.close()

    # Annual mean, fill masked values with median
    def _annual_mean(arr):
        if hasattr(arr, "filled"):
            arr = arr.filled(np.nan)
        mean = np.nanmean(arr.astype(np.float32), axis=0)
        median = float(np.nanmedian(mean))
        mean = np.where(np.isnan(mean), median, mean)
        return mean

    emis_b07 = _annual_mean(emis_b07_monthly)
    emis_b14 = _annual_mean(emis_b14_monthly)

    # Regrid to ABI target grid
    target_lats, target_lons = _linspace_grid(region_cfg, shape)

    def _regrid_emis(field, src_lats, src_lons):
        # RegularGridInterpolator needs ascending lats
        if src_lats[0] > src_lats[-1]:
            src_lats = src_lats[::-1]
            field    = field[::-1, :]
        interp = RegularGridInterpolator(
            (src_lats, src_lons), field,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        # Build query meshgrid (target_lats may be descending — handle it)
        tl_asc = np.sort(target_lats)
        lon_g, lat_g = np.meshgrid(target_lons, tl_asc)
        pts    = np.column_stack([lat_g.ravel(), lon_g.ravel()])
        result = interp(pts).reshape(len(tl_asc), len(target_lons))
        # Flip back so row 0 = north
        if target_lats[0] > target_lats[-1]:
            result = result[::-1, :]
        return result.astype(np.float32)

    print("  🔄 Regriding emissivity B07 ...", end=" ", flush=True)
    e07 = _regrid_emis(emis_b07, lats_global, lons_global)
    np.save(out_b07, e07)
    print(f"saved {e07.shape}")

    print("  🔄 Regriding emissivity B14 ...", end=" ", flush=True)
    e14 = _regrid_emis(emis_b14, lats_global, lons_global)
    np.save(out_b14, e14)
    print(f"saved {e14.shape}")


# ── Land / sea mask ───────────────────────────────────────────────────────────

def _build_land_sea_mask(region_cfg: dict, shape: tuple[int, int], static_dir: str):
    """
    Rasterise the GSHHG full-resolution land polygons to a binary mask.
    1 = land, 0 = sea.  Saved as int8.
    """
    import fiona
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import shape as shapely_shape, box

    out_path = os.path.join(static_dir, "land_sea_mask.npy")
    if os.path.exists(out_path):
        print("  ✅ Land/sea mask already exists, skipping.")
        return

    rows, cols = shape
    bbox = box(
        region_cfg["lon_min"], region_cfg["lat_min"],
        region_cfg["lon_max"], region_cfg["lat_max"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "gshhg.zip")
        _download(_GSHHG_URL, zip_path, "GSHHG land polygons (zip)")

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        # Full-resolution land boundary: GSHHS_f_L1.shp
        shp_path = None
        for root, _, files in os.walk(tmp):
            for f in files:
                if f == "GSHHS_f_L1.shp":
                    shp_path = os.path.join(root, f)
                    break

        if shp_path is None:
            raise FileNotFoundError("GSHHS_f_L1.shp not found in GSHHG zip")

        print("  🔄 Rasterising land polygons ...", end=" ", flush=True)

        transform = from_bounds(
            region_cfg["lon_min"], region_cfg["lat_min"],
            region_cfg["lon_max"], region_cfg["lat_max"],
            cols, rows,
        )

        geoms = []
        with fiona.open(shp_path) as src:
            for feature in src:
                geom = shapely_shape(feature["geometry"])
                clipped = geom.intersection(bbox)
                if not clipped.is_empty:
                    geoms.append(clipped)

        mask = rasterize(
            [(g, 1) for g in geoms],
            out_shape=(rows, cols),
            transform=transform,
            fill=0,
            dtype=np.int8,
        )

    np.save(out_path, mask)
    print(f"saved {mask.shape}  (land pixels: {mask.sum():,})")


# ── Land cover ────────────────────────────────────────────────────────────────

def _build_land_cover(region_cfg: dict, shape: tuple[int, int], static_dir: str):
    """
    Crop ESA GlobCover 2009 to the region and regrid to ABI shape.
    Saved as int16 (GlobCover class codes).
    """
    import rasterio
    from rasterio.windows import from_bounds as window_from_bounds
    from scipy.ndimage import zoom

    out_path = os.path.join(static_dir, "land_cover.npy")
    if os.path.exists(out_path):
        print("  ✅ Land cover already exists, skipping.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "globcover.zip")
        _download(_GLOBCOVER_URL, zip_path, "ESA GlobCover 2009 (zip)")

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        # Find the GeoTIFF inside the zip
        tif_path = None
        for root, _, files in os.walk(tmp):
            for f in files:
                if f.endswith(".tif") and "GLOBCOVER" in f.upper():
                    tif_path = os.path.join(root, f)
                    break

        if tif_path is None:
            raise FileNotFoundError("GlobCover GeoTIFF not found in zip")

        print("  🔄 Cropping land cover ...", end=" ", flush=True)

        with rasterio.open(tif_path) as src:
            window = window_from_bounds(
                region_cfg["lon_min"], region_cfg["lat_min"],
                region_cfg["lon_max"], region_cfg["lat_max"],
                src.transform,
            )
            crop = src.read(1, window=window)  # int16

    rows, cols = shape
    zoom_r = rows / crop.shape[0]
    zoom_c = cols / crop.shape[1]
    resampled = zoom(crop, (zoom_r, zoom_c), order=0).astype(np.int16)  # nearest neighbour

    np.save(out_path, resampled)
    print(f"saved {resampled.shape}")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_all(region_cfg: dict, output_root: str, goes_shape: tuple[int, int] | None = None):
    """
    Download and prepare all static files for a region.

    Parameters
    ----------
    region_cfg  : dict with lat_min/lat_max/lon_min/lon_max
    output_root : pipeline output root for this region (used to infer ABI shape)
    goes_shape  : override ABI grid shape; if None, inferred from saved ABI files
    """
    os.makedirs(STATIC_DIR, exist_ok=True)

    shape = goes_shape or _infer_goes_shape(output_root)
    if shape is None:
        # Fallback: 0.02° grid (~2 km) over the bounding box
        rows = int(round((region_cfg["lat_max"] - region_cfg["lat_min"]) / 0.02))
        cols = int(round((region_cfg["lon_max"] - region_cfg["lon_min"]) / 0.02))
        shape = (rows, cols)
        print(f"  ⚠  No ABI files found to infer grid shape — using {shape} fallback")

    print("\n📦 Building static ancillary data")
    print(f"   Region : lat [{region_cfg['lat_min']}, {region_cfg['lat_max']}]  "
          f"lon [{region_cfg['lon_min']}, {region_cfg['lon_max']}]")
    print(f"   Shape  : {shape}")
    print(f"   Output : {os.path.abspath(STATIC_DIR)}\n")

    print("── Emissivities ──────────────────────────────────────────────────────")
    _build_emissivity(region_cfg, shape, STATIC_DIR)

    print("\n── Land / sea mask ───────────────────────────────────────────────────")
    _build_land_sea_mask(region_cfg, shape, STATIC_DIR)

    print("\n── Land cover ────────────────────────────────────────────────────────")
    _build_land_cover(region_cfg, shape, STATIC_DIR)

    print("\n✅ All static files ready.\n")


def load_static(static_dir: str = STATIC_DIR) -> dict:
    """
    Load all static arrays into a dict for use in fire_detection.py.

    Returns
    -------
    {
      "emissivity_b07" : np.ndarray float32,
      "emissivity_b14" : np.ndarray float32,
      "land_sea_mask"  : np.ndarray int8,
      "land_cover"     : np.ndarray int16,
    }
    """
    files = {
        "emissivity_b07": "emissivity_b07.npy",
        "emissivity_b14": "emissivity_b14.npy",
        "land_sea_mask":  "land_sea_mask.npy",
        "land_cover":     "land_cover.npy",
    }
    result = {}
    for key, fname in files.items():
        path = os.path.join(static_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Static file missing: {path}\n"
                "Run static_data_downloader.build_all() first."
            )
        result[key] = np.load(path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download static ancillary data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--region", default=None, help="Region key in config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    region_key = args.region or next(iter(cfg["regions"]))
    region_cfg = cfg["regions"][region_key]
    output_root = os.path.join(cfg["output_root"], region_key)

    build_all(region_cfg, output_root)