"""
FDCA – Fire Detection and Characterization Algorithm
Main entry point. Orquestra Part I y Part II.
Flujo:
FDCAInput
    ↓
run_part1()
    ↓
candidatos de fuego
    ↓
run_part2()
    ↓
confirmación temporal/refinamiento
    ↓
FDCAOutput
Uso:
-----
>>> from fdca import run_fdca, FDCAInput, FDCAOutput
>>> result = run_fdca(inputs)
>>> result.fire_mask      # 2-D int array of FireMask codes
>>> result.confirmed_fires  # list of FireCandidate with Tt, p, FRP, ...
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

from .part1 import run_part1, FireCandidate
from .part2 import run_part2
from .constants import FireMask


# ── Contenedor de todas las entradas del algoritmo ───────────────────────────
@dataclass
class FDCAInput:
    """All inputs required by the FDCA (Tables 3.1, 3.2, 3.3 of ATBD)."""

    # ── ABI band data (2-D arrays, shape [L, W]) ─────────────────────────────
    bt7:   np.ndarray              # Ch 7 brightness temperature [K]
    rad7:  np.ndarray              # Ch 7 radiance [W·m⁻²·sr⁻¹·m⁻¹]
    bt14:  np.ndarray              # Ch 14 brightness temperature [K]
    rad14: np.ndarray              # Ch 14 radiance
    bt13:  Optional[np.ndarray]    # Ch 13 BT (needed if FPT > 90 K)
    rad13: Optional[np.ndarray]    # Ch 13 radiance
    bt15:  Optional[np.ndarray]    # Ch 15 BT (optional)
    rad15: Optional[np.ndarray]    # Ch 15 radiance (optional)
    refl2: Optional[np.ndarray]    # Ch 2 reflectance (0–1, optional)

    # ── Geolocation / geometry ───────────────────────────────────────────────
    latitudes:   np.ndarray        # [deg]
    longitudes:  np.ndarray        # [deg]
    sza:         np.ndarray        # Solar zenith angle [deg]
    glint_angle: np.ndarray        # Sun glint angle [deg]
    lza:         np.ndarray        # Local (satellite) zenith angle [deg]
    azimuth:     np.ndarray        # Relative azimuth [deg]

    # ── Dynamic ancillary ────────────────────────────────────────────────────
    tpw:     np.ndarray            # Total Precipitable Water [mm]
    emiss7:  np.ndarray            # Surface emissivity at 3.9 µm
    emiss14: np.ndarray            # Surface emissivity at 11.2 µm
    lut_tpw: np.ndarray            # TPW LUT (6 rows × 35 columns)
    FPT:     float                 # Focal Plane Temperature [K]

    # ── Static ancillary ─────────────────────────────────────────────────────
    land_cover:  np.ndarray        # MODIS/UMD land cover class
    land_mask:   np.ndarray        # Boolean (True = land)
    desert_mask: np.ndarray        # Desert mask (2 = bright desert)
    usgs_eco:    np.ndarray        # USGS ecosystem type

    # ── Temporal filtering ───────────────────────────────────────────────────
    prev_fire_mask: Optional[np.ndarray] = None  # seconds-since-2001 of last fire
    scan_time:      Optional[datetime]   = None  # UTC time of current scan

    # ── Planck calibration coefficients (per-band, per-timestamp) ───────────
    coeffs7:  Optional[dict] = None   # {"fk1":.., "fk2":.., "bc1":.., "bc2":..}
    coeffs14: Optional[dict] = None
    coeffs13: Optional[dict] = None
    coeffs15: Optional[dict] = None

    # ── Optional data quality flags ──────────────────────────────────────────
    data_quality: Optional[np.ndarray] = None


# ── Output container ──────────────────────────────────────────────────────────
@dataclass
class FDCAOutput:
    """Algorithm outputs (Table 3.10 of ATBD)."""

    fire_mask:     np.ndarray              # Per-pixel mask codes (Table 3.11)
    fail_char_arr: np.ndarray              # FailChar codes per pixel
    confirmed_fires: List[FireCandidate]   # Only Part-II confirmed fires

    # ── Scalar metadata ───────────────────────────────────────────────────────
    n_candidates:   int = 0
    n_confirmed:    int = 0
    n_processed:    int = 0   # mask code 10 or 30
    n_saturated:    int = 0   # 11 / 31
    n_cloudy:       int = 0   # 12 / 32
    n_high:         int = 0   # 13 / 33
    n_medium:       int = 0   # 14 / 34
    n_low:          int = 0   # 15 / 35

    def summary(self) -> str:
        lines = [
            "=== FDCA Output Summary ===",
            f"  Part I candidates : {self.n_candidates}",
            f"  Part II confirmed : {self.n_confirmed}",
            f"  Processed  (10/30): {self.n_processed}",
            f"  Saturated  (11/31): {self.n_saturated}",
            f"  Cloudy     (12/32): {self.n_cloudy}",
            f"  High prob  (13/33): {self.n_high}",
            f"  Medium prob(14/34): {self.n_medium}",
            f"  Low prob   (15/35): {self.n_low}",
        ]
        return "\n".join(lines)


# ── Reference epoch for temporal filtering ────────────────────────────────────
_EPOCH_2001 = datetime(2001, 1, 1, 0, 0, 0)


def _to_epoch(dt: Optional[datetime]) -> float:
    if dt is None:
        return 0.0
    return (dt - _EPOCH_2001).total_seconds()


# ── Main entry point ──────────────────────────────────────────────────────────
def run_fdca(inp: FDCAInput) -> FDCAOutput:
    """
    Run the full FDCA algorithm (Part I + Part II).

    Parameters
    ----------
    inp : FDCAInput

    Returns
    -------
    FDCAOutput
    """
    # ── Part I ────────────────────────────────────────────────────────────────
    fire_mask, fail_char_arr, candidates = run_part1(
        bt7=inp.bt7, rad7=inp.rad7,
        bt14=inp.bt14, rad14=inp.rad14,
        bt13=inp.bt13, rad13=inp.rad13,
        bt15=inp.bt15, refl2=inp.refl2,
        latitudes=inp.latitudes, longitudes=inp.longitudes,
        sza=inp.sza, glint_angle=inp.glint_angle,
        lza=inp.lza, azimuth=inp.azimuth,
        tpw=inp.tpw, emiss7=inp.emiss7, emiss14=inp.emiss14,
        lut_tpw=inp.lut_tpw, FPT=inp.FPT,
        coeffs7=inp.coeffs7, coeffs14=inp.coeffs14, coeffs13=inp.coeffs13,
        land_cover=inp.land_cover, land_mask=inp.land_mask,
        desert_mask=inp.desert_mask, usgs_eco=inp.usgs_eco,
        data_quality=inp.data_quality,
    )

    # ── Part II ───────────────────────────────────────────────────────────────
    current_epoch = _to_epoch(inp.scan_time)
    fire_mask, fail_char_arr, confirmed = run_part2(
        candidates=candidates,
        fire_mask=fire_mask,
        fail_char_arr=fail_char_arr,
        prev_fire_mask=inp.prev_fire_mask,
        current_epoch=current_epoch,
    )

    # ── Tally output statistics ───────────────────────────────────────────────
    processed = saturated = cloudy = high = medium = low = 0
    for f in confirmed:
        code = fire_mask[f.i, f.j]
        base = code if code < 30 else code - 20
        if   base == FireMask.PROCESSED:   processed += 1
        elif base == FireMask.SATURATED:   saturated += 1
        elif base == FireMask.CLOUD_CONTAM: cloudy  += 1
        elif base == FireMask.HIGH_PROB:   high     += 1
        elif base == FireMask.MED_PROB:    medium   += 1
        elif base == FireMask.LOW_PROB:    low      += 1

    return FDCAOutput(
        fire_mask=fire_mask,
        fail_char_arr=fail_char_arr,
        confirmed_fires=confirmed,
        n_candidates=len(candidates),
        n_confirmed=len(confirmed),
        n_processed=processed,
        n_saturated=saturated,
        n_cloudy=cloudy,
        n_high=high,
        n_medium=medium,
        n_low=low,
    )
