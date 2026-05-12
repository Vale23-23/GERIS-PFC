"""
tpw_downloader.py — Download and regrid GFS Total Precipitable Water (TPW)
to match the GOES ABI grid for a given region and timestamp.

The GFS publishes analyses every 6 hours (00, 06, 12, 18 UTC) on the NOMADS
server in GRIB2 format. This module:
  1. Rounds a given timestamp to the nearest GFS cycle.
  2. Downloads only the PWAT layer via the NOMADS HTTP filter (no full GRIB).
  3. Crops to the region bounding box.
  4. Regrids from GFS ~25 km resolution to the ABI pixel grid.
  5. Saves as .npy (float32) with the same timestamp naming as downloader.py.

Return dict always matches downloader.download_and_save() schema:
  {
    "status":    "downloaded" | "exists" | "empty" | "error",
    "path":      str | None,
    "product":   "TPW-GFS",
    "timestamp": "YYYYMMDD_HHMM",
    "shape":     [rows, cols],   # only on success
    "error":     str,            # only on error
  }

Dependencies (add to requirements.txt):
  cfgrib          # reads GRIB2 with xarray
  eccodes         # cfgrib backend (system package or conda)
  scipy           # RegularGridInterpolator
  requests        # HTTP download
"""

import os
import re
import tempfile
from datetime import datetime, timedelta

import numpy as np
import requests

PRODUCT_ID = "TPW-GFS"

# ── NOMADS URL template ───────────────────────────────────────────────────────
# filter_gfs.pl lets us request a single variable/level so we don't download
# the full ~500 MB GRIB file.
# Docs: https://nomads.ncep.noaa.gov/txt_descriptions/GRIB_filter_doc.shtml
_NOMADS_URL = (
    "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    "?dir=%2Fgfs.{date}%2F{cycle:02d}%2Fatmos"
    "&file=gfs.t{cycle:02d}z.pgrb2.0p25.anl"
    "&var_PWAT=on"
    "&lev_entire_atmosphere=on"
    "&subregion=&leftlon={lon_min}&rightlon={lon_max}"
    "&toplat={lat_max}&bottomlat={lat_min}"
)

_TIMEOUT_SEC = 60


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nearest_gfs_cycle(ts: datetime) -> tuple[datetime, int]:
    """
    Round ts down to the nearest GFS analysis cycle (00/06/12/18 UTC).
    Returns (cycle_datetime, cycle_hour).
    """
    cycle_hour = (ts.hour // 6) * 6
    cycle_dt   = ts.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    return cycle_dt, cycle_hour


def _build_url(cycle_dt: datetime, cycle_hour: int, region_cfg: dict) -> str:
    return _NOMADS_URL.format(
        date    = cycle_dt.strftime("%Y%m%d"),
        cycle   = cycle_hour,
        lon_min = region_cfg["lon_min"],
        lon_max = region_cfg["lon_max"],
        lat_min = region_cfg["lat_min"],
        lat_max = region_cfg["lat_max"],
    )


def _download_grib(url: str) -> str:
    """
    Download the filtered GRIB2 to a temp file and return its path.
    Raises requests.HTTPError on non-200 responses.
    """
    resp = requests.get(url, timeout=_TIMEOUT_SEC, stream=True)
    resp.raise_for_status()

    # Save to a named temp file (cfgrib needs a real path, not a BytesIO)
    tmp = tempfile.NamedTemporaryFile(suffix=".grb2", delete=False)
    for chunk in resp.iter_content(chunk_size=1 << 16):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def _parse_pwat(grib_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Open the GRIB2 file with cfgrib and extract PWAT.
    Returns (pwat_2d, lats_1d, lons_1d) — lats descending, lons ascending.
    PWAT unit in GFS is kg/m² ≡ mm of precipitable water.
    """
    import cfgrib  # lazy import so the rest of the module works without it

    ds = cfgrib.open_dataset(
        grib_path,
        filter_by_keys={"shortName": "pwat"},
        indexpath=None,   # don't write .idx files next to the temp file
    )
    pwat = ds["pwat"].values.astype(np.float32)   # shape (lat, lon)
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    return pwat, lats, lons


def _build_goes_latlon_grid(goes_npy_shape, region_cfg):
    """
    Build a regular lat/lon grid that matches the saved .npy ABI array.

    The ABI crop is saved as a plain 2-D array aligned to the bounding box,
    so we reconstruct the grid by linspace over the bounding box.
    Row 0 = north (lat_max), last row = south (lat_min) — same as origin='upper'.
    """
    rows, cols = goes_npy_shape
    lats = np.linspace(region_cfg["lat_max"], region_cfg["lat_min"], rows)
    lons = np.linspace(region_cfg["lon_min"], region_cfg["lon_max"], cols)
    return lats, lons   # 1-D arrays


def _regrid(pwat, gfs_lats, gfs_lons, target_lats, target_lons):
    """
    Bilinear interpolation from the GFS grid to the ABI pixel grid.
    Uses scipy.interpolate.RegularGridInterpolator.

    GFS lats may be descending; RegularGridInterpolator requires ascending,
    so we flip if needed.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Ensure lats are ascending for the interpolator
    if gfs_lats[0] > gfs_lats[-1]:
        gfs_lats = gfs_lats[::-1]
        pwat     = pwat[::-1, :]

    interp = RegularGridInterpolator(
        (gfs_lats, gfs_lons),
        pwat,
        method       = "linear",
        bounds_error = False,
        fill_value   = np.nan,
    )

    # Build meshgrid of target points (rows = lats descending → we reverse)
    target_lats_asc = target_lats[::-1]   # ascending for meshgrid query
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats_asc)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    result = interp(points).reshape(len(target_lats_asc), len(target_lons))
    return result[::-1, :].astype(np.float32)   # flip back to north-first


def _infer_goes_shape(output_root: str, timestamp: datetime) -> tuple[int, int] | None:
    """
    Try to read the shape of an already-saved ABI band for this timestamp
    so we can match the regrid target exactly.
    Returns (rows, cols) or None if not found.
    """
    ts_str = timestamp.strftime("%Y%m%d_%H%M")
    # Try the most common band first, then others
    for product in ("ABI-L1b-Rad-B07", "ABI-L1b-Rad-B14", "ABI-L1b-Rad-B13"):
        path = os.path.join(output_root, product, f"{ts_str}.npy")
        if os.path.exists(path):
            arr = np.load(path, mmap_mode="r")
            return arr.shape
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def download_and_save(
    timestamp:   datetime,
    region_cfg:  dict,
    output_root: str,
    goes_shape:  tuple[int, int] | None = None,
) -> dict:
    """
    Download GFS TPW for the given timestamp, regrid to the ABI grid, save as .npy.

    Parameters
    ----------
    timestamp   : datetime — target time (UTC)
    region_cfg  : dict     — must have lat_min/lat_max/lon_min/lon_max keys
    output_root : str      — root folder (same as pipeline output_root + region)
    goes_shape  : (rows, cols) | None
                  Shape of the ABI arrays for this region. If None, the module
                  tries to infer it from already-saved ABI bands; if that also
                  fails it falls back to a linspace grid at GFS resolution.

    Returns
    -------
    dict with status / path / product / timestamp  (mirrors downloader.py schema)
    """
    ts_str     = timestamp.strftime("%Y%m%d_%H%M")
    folder     = os.path.join(output_root, PRODUCT_ID)
    os.makedirs(folder, exist_ok=True)
    file_path  = os.path.join(folder, f"{ts_str}.npy")

    if os.path.exists(file_path):
        return {"status": "exists", "path": file_path,
                "product": PRODUCT_ID, "timestamp": ts_str}

    cycle_dt, cycle_hour = _nearest_gfs_cycle(timestamp)
    url = _build_url(cycle_dt, cycle_hour, region_cfg)

    grib_path = None
    try:
        # 1. Download filtered GRIB2
        grib_path = _download_grib(url)

        # 2. Parse PWAT
        pwat, gfs_lats, gfs_lons = _parse_pwat(grib_path)

        if pwat.size == 0:
            return {"status": "empty", "path": None,
                    "product": PRODUCT_ID, "timestamp": ts_str}

        # 3. Determine target grid shape
        shape = goes_shape or _infer_goes_shape(output_root, timestamp)

        if shape is not None:
            target_lats, target_lons = _build_goes_latlon_grid(shape, region_cfg)
            pwat_regridded = _regrid(pwat, gfs_lats, gfs_lons, target_lats, target_lons)
        else:
            # Fallback: save at native GFS resolution (no regrid needed)
            pwat_regridded = pwat

        # 4. Save
        np.save(file_path, pwat_regridded)

        return {
            "status":    "downloaded",
            "path":      file_path,
            "product":   PRODUCT_ID,
            "timestamp": ts_str,
            "shape":     list(pwat_regridded.shape),
            "gfs_cycle": cycle_dt.strftime("%Y%m%d_%H%M"),
        }

    except requests.HTTPError as e:
        return {
            "status":    "error",
            "substatus": "error_nomads",
            "path":      None,
            "product":   PRODUCT_ID,
            "timestamp": ts_str,
            "error":     f"NOMADS HTTP error: {e}  URL: {url}",
        }

    except Exception as e:
        return {
            "status":    "error",
            "substatus": "error_local",
            "path":      None,
            "product":   PRODUCT_ID,
            "timestamp": ts_str,
            "error":     str(e),
        }

    finally:
        # Always clean up the temp GRIB file
        if grib_path and os.path.exists(grib_path):
            os.remove(grib_path)