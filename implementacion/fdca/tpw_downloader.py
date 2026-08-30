"""
tpw_downloader.py — Download and regrid GFS Total Precipitable Water (TPW)
to match the GOES ABI grid for a given region and timestamp.

The GFS publishes analyses every 6 hours (00, 06, 12, 18 UTC). This module
downloads them from the NOAA Big Data Program's public AWS S3 archive
(bucket noaa-gfs-bdp-pds), not the NOMADS operational filter service:
NOMADS only retains a short rolling window (~10 days) of recent cycles,
so historical timestamps (e.g. this project's fixed test scene, from
2025) are not available there. This module:
  1. Rounds a given timestamp to the nearest GFS cycle.
  2. Fetches the .idx sidecar to locate the PWAT message's byte offsets,
     then downloads only that byte range via an HTTP Range request
     (S3 has no server-side variable filter like NOMADS did, so this
     replaces that functionality client-side).
  3. Crops the resulting full-globe field to the region bounding box.
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

_S3_GRIB_URL = (
    "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    "/gfs.{date}/{cycle:02d}/atmos"
    "/gfs.t{cycle:02d}z.pgrb2.0p25.f000"
)

# GRIB2 .idx variable/level identifiers as published by NOAA (NCO product
# reference: nco.ncep.noaa.gov/pmb/products/gfs/gfs.t00z.pgrb2.0p25.f000.shtml).
# Used to locate the PWAT byte range inside the multi-message .idx sidecar
# file without downloading the full ~500 MB GRIB2.
_IDX_PWAT_VAR   = "PWAT"
_IDX_PWAT_LEVEL = "entire atmosphere (considered as a single layer)"

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

def _build_grib_url(cycle_dt: datetime, cycle_hour: int) -> str:
    """
    Build the S3 URL of the full-globe GFS GRIB2 file for a given cycle.

    Unlike the retired NOMADS filter endpoint, S3 serves the raw archive
    file with no server-side variable/subregion filtering, so region_cfg
    is not needed here -- cropping happens locally after the byte-range
    download (see _crop_to_region).
    """
    return _S3_GRIB_URL.format(
        date  = cycle_dt.strftime("%Y%m%d"),
        cycle = cycle_hour,
    )


def _parse_idx_byte_range(idx_text: str, var: str, level: str) -> tuple[int, int | None]:
    """
    Locate the byte range of a single variable/level inside a GRIB2 .idx
    sidecar file.

    Each line of the .idx has the form:
        <message_number>:<byte_offset>:d=<YYYYMMDDHH>:<VAR>:<LEVEL>:<fcst>:
    Byte offsets mark the start of each GRIB2 message (each of which is
    itself a complete, independently-decodable GRIB2 file beginning with
    "GRIB" and ending with "7777"). The end of the target message is the
    start offset of the next message minus one, or "until EOF" if it is
    the last message in the file.

    Returns
    -------
    (start_byte, end_byte) -- end_byte is None when the range should
    extend to the end of the file (open-ended HTTP Range request).

    Raises
    ------
    ValueError if var/level is not found in the index.
    """
    offsets: list[int] = []
    match_idx: int | None = None

    for pos, line in enumerate(idx_text.strip().splitlines()):
        fields = line.split(":")
        if len(fields) < 5:
            continue
        offsets.append(int(fields[1]))
        if fields[3] == var and fields[4] == level:
            match_idx = pos

    if match_idx is None:
        raise ValueError(f"Variable '{var}' / level '{level}' not found in .idx")

    start = offsets[match_idx]
    end   = offsets[match_idx + 1] - 1 if match_idx + 1 < len(offsets) else None
    return start, end


def _download_grib_byte_range(grib_url: str, start: int, end: int | None) -> str:
    """
    Download only the PWAT message from the S3 GRIB2 archive file, via an
    HTTP Range request, using the byte offsets found in the .idx file.
    Raises requests.HTTPError on non-2xx responses.
    """
    range_header = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
    resp = requests.get(
        grib_url, headers={"Range": range_header}, timeout=_TIMEOUT_SEC, stream=True
    )
    resp.raise_for_status()

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

def _crop_to_region(
    pwat: np.ndarray, lats: np.ndarray, lons: np.ndarray, region_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop the global GFS PWAT field to the region bounding box.

    Needed because S3 (unlike the retired NOMADS filter) has no server-side
    subregion cropping -- _parse_pwat returns the full-globe 1440x721 grid.

    GFS longitudes are in [0, 360); region_cfg and the rest of the
    pipeline (e.g. downloader.py's ABI bounding boxes, _build_goes_latlon_grid)
    use [-180, 180]. The comparison against region_cfg is done in [0, 360]
    to match the GFS grid, but the returned lons are converted BACK to
    [-180, 180] -- otherwise _regrid ends up querying an interpolator
    built on a [0,360)-domain grid with [-180,180) target points, and every
    single query falls "out of bounds", silently returning NaN everywhere.
    """
    lon_min_360 = region_cfg["lon_min"] % 360.0
    lon_max_360 = region_cfg["lon_max"] % 360.0
    lat_min = region_cfg["lat_min"]
    lat_max = region_cfg["lat_max"]

    lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lon_idx = np.where((lons >= lon_min_360) & (lons <= lon_max_360))[0]

    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError(
            f"Region bounding box (lat [{lat_min}, {lat_max}], "
            f"lon [{lon_min_360}, {lon_max_360}] after 0-360 conversion) "
            f"does not intersect the GFS grid"
        )

    pwat_crop = pwat[np.ix_(lat_idx, lon_idx)]
    lats_crop = lats[lat_idx]
    # Convert back to [-180, 180] to match region_cfg / target_lons downstream.
    # (Doesn't handle regions straddling the 180° meridian -- not a concern
    # for Uruguay, but worth a comment if this module is reused elsewhere.)
    lons_crop = ((lons[lon_idx] + 180.0) % 360.0) - 180.0

    return pwat_crop, lats_crop, lons_crop

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
def dedupe_by_cycle(timestamps: list[datetime]) -> list[datetime]:
    """
    Collapse a batch of ABI scene timestamps down to the distinct GFS cycles
    they fall into.

    Callers that dispatch TPW downloads for many scenes (e.g. a pipeline
    processing a whole day) must download_and_save() once per returned
    cycle, not once per input timestamp -- otherwise concurrent calls for
    timestamps sharing a cycle race on the same output file, each seeing
    "does not exist yet" and re-downloading it.
    """
    seen: dict[str, datetime] = {}
    for ts in timestamps:
        cycle_dt, _ = _nearest_gfs_cycle(ts)
        seen.setdefault(cycle_dt.strftime("%Y%m%d_%H%M"), cycle_dt)
    return sorted(seen.values())

def cycle_path_for_timestamp(timestamp: datetime, output_root: str) -> str:
    """
    Path of the GFS TPW file covering a given ABI timestamp.

    Readers (e.g. fdca_adapter.get_tpw_real) must resolve the same nearest-
    6h-cycle file that download_and_save() writes to, since TPW-GFS/*.npy
    is no longer keyed by ABI timestamp.
    """
    cycle_dt, _ = _nearest_gfs_cycle(timestamp)
    cycle_str = cycle_dt.strftime("%Y%m%d_%H%M")
    return os.path.join(output_root, PRODUCT_ID, f"{cycle_str}.npy")

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
    ts_str = timestamp.strftime("%Y%m%d_%H%M")
    folder = os.path.join(output_root, PRODUCT_ID)
    os.makedirs(folder, exist_ok=True)

    # GFS only publishes one analysis every 6h (00/06/12/18 UTC), so every ABI
    # scene falling in that window (up to ~36 scenes at 10-min cadence) maps to
    # the exact same GFS field. Cache by cycle, not by ABI timestamp — otherwise
    # every scene re-downloads and re-saves an identical file.
    cycle_dt, cycle_hour = _nearest_gfs_cycle(timestamp)
    cycle_str = cycle_dt.strftime("%Y%m%d_%H%M")
    file_path = os.path.join(folder, f"{cycle_str}.npy")

    if os.path.exists(file_path):
        return {"status": "exists", "path": file_path,
                "product": PRODUCT_ID, "timestamp": ts_str,
                "gfs_cycle": cycle_str}

    grib_url = _build_grib_url(cycle_dt, cycle_hour)

    grib_path = None
    try:
        # 1. Fetch the .idx sidecar and locate the PWAT byte range
        idx_resp = requests.get(grib_url + ".idx", timeout=_TIMEOUT_SEC)
        idx_resp.raise_for_status()
        start, end = _parse_idx_byte_range(idx_resp.text, _IDX_PWAT_VAR, _IDX_PWAT_LEVEL)

        # 2. Download only that byte range (a few KB, not the full ~500 MB file)
        grib_path = _download_grib_byte_range(grib_url, start, end)

        # 3. Parse PWAT (full-globe grid -- S3 does no server-side cropping)
        pwat, gfs_lats, gfs_lons = _parse_pwat(grib_path)

        if pwat.size == 0:
            return {"status": "empty", "path": None,
                    "product": PRODUCT_ID, "timestamp": ts_str}

        # 4. Crop to the region bounding box
        pwat, gfs_lats, gfs_lons = _crop_to_region(pwat, gfs_lats, gfs_lons, region_cfg)

        # 5. Determine target grid shape
        shape = goes_shape or _infer_goes_shape(output_root, timestamp)

        if shape is not None:
            target_lats, target_lons = _build_goes_latlon_grid(shape, region_cfg)
            pwat_regridded = _regrid(pwat, gfs_lats, gfs_lons, target_lats, target_lons)
        else:
            # Fallback: save at native GFS resolution (no regrid needed)
            pwat_regridded = pwat

        # 6. Save
        np.save(file_path, pwat_regridded)

        return {
            "status":    "downloaded",
            "path":      file_path,
            "product":   PRODUCT_ID,
            "timestamp": ts_str,
            "shape":     list(pwat_regridded.shape),
            "gfs_cycle": cycle_str,
        }

    except requests.HTTPError as e:
        return {
            "status":    "error",
            "substatus": "error_s3",
            "path":      None,
            "product":   PRODUCT_ID,
            "timestamp": ts_str,
            "error":     f"GFS S3 archive HTTP error: {e}  URL: {grib_url}",
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