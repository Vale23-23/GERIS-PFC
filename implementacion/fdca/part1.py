"""
FDCA Part I – Loop over all pixels
Implements ATBD sections 3.4.2.1 through 3.4.2.13
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field

from .constants import (
    FireMask, FailChar,
    SAT_TEMP_CH7, SAT_TEMP_CH14, SAT_BUFF,
    SAT_FLAG_CH7, SAT_FLAG_CH14,
    FPT_THRESHOLD, MIN_BT,
    MAX_LOCAL_ZENITH, MAX_SZA_DAYLIGHT, GLINT_THRESHOLD,
    CLOUD_BT14_THRESH, CLOUD_BT7_BT14_NEG, CLOUD_BT7_BT14_POS,
    CLOUD_BT7_FOR_POS, CLOUD_ALBEDO_THRESH,
    CLOUD_BT15_THRESH, CLOUD_BT14_BT15_NEG, CLOUD_BT14_BT15_POS,
    BT7_MIN_NIGHT, BT7_MIN_SOLAR_COEF,
    BT7_REFL_THRESH_NIGHT, BT7_REFL_THRESH_SOLAR,
    CLOUD_ADJ_ALBEDO_LOW, CLOUD_ADJ_ALBEDO_HIGH,
    CLOUD_ADJ_BT7_COEF, CLOUD_ADJ_BT14_COEF,
    CLOUD_ADJ_BT7_FIXED, CLOUD_ADJ_BT14_FIXED,
    DIFFRAC_CH7_SUB, DIFFRAC_CH7_DIV, DIFFRAC_CH14_SUB, DIFFRAC_CH14_DIV,
    MIN_FIRE_TEMP, MAX_SURF_TEMP, MIN_PIXEL_AREA,
    MODIS_WATER_CODES, UMD_WATER_CODE, DESERT_BRIGHT,
    USGS_SEA_WATER, USGS_COAST_FRINGE, USGS_INLAND_WATER,
)
from .planck import planck_rad, planck_temp, temp_to_rad_in_band
from .background import BackgroundStats, compute_background
from .dozier import DozierResult, compute_dozier, compute_frp, compute_pixel_area


# ── Per-pixel fire candidate record (passed to Part II) ──────────────────────
@dataclass
class FireCandidate:
    i: int
    j: int
    lat: float
    lon: float
    fire_id: int

    # Observed values
    bt7:  float = np.nan
    bt14: float = np.nan
    rad7: float = np.nan
    rad14: float = np.nan

    # Corrected values
    bt7_corr:  float = np.nan
    bt14_corr: float = np.nan
    bt_bkg_corr: float = np.nan

    # Background
    bt7_bkg:  float = np.nan
    bt14_bkg: float = np.nan
    bt7_bkg_std:  float = np.nan
    bt14_bkg_std: float = np.nan
    reflb:       float = np.nan
    rad_diff_sigma: float = np.nan
    n_passes: int = 0

    # Thresholds
    std_dev_7b_14b: float = np.nan
    std_dev_7b:     float = np.nan
    std_dev_reflb:  float = np.nan
    std_dev_reflb_max: float = np.nan

    # Dozier
    fire_temp: float = np.nan
    fire_frac: float = np.nan
    fire_area: float = np.nan
    frp:       float = -99.0

    # Flags
    fail_char:   int  = FailChar.NONE
    is_saturated: bool = False
    is_cloudy:    bool = False
    is_day:       bool = False

    # Geometry
    sza:   float = np.nan
    lza:   float = np.nan
    azimuth: float = np.nan

    # Visible
    albedo:     float = np.nan
    albedo_bkg: float = np.nan
    vis_count:  float = np.nan
    vis_bkg:    float = np.nan

    # Emissivity
    emiss7:  float = 1.0
    emiss14: float = 1.0

    # Refl product for pixel
    refl_pixel: float = np.nan
    refl_m2:    float = np.nan   # i-2 along scan
    refl_p2:    float = np.nan   # i+2 along scan
    pass_along_scan: bool = False

    # Background full stats object
    bkg: Optional[BackgroundStats] = None


# ── Helper: contextual thresholds (ATBD 3.4.2.6) ────────────────────────────
def _contextual_thresholds(bkg: BackgroundStats, sza_cos: float):
    """
    Returns (std_7b14b, std_7b, std_reflb, std_reflb_max, pass_along_scan_fn)
    """
    n_pass = bkg.n_passes

    # Std Dev (Tb3.9 - Tb11.2) test  →  max 4.0
    std_7b14b = min(4.0, 3.0 * bkg.std_dev_7_14_diff)

    # Std Dev (Tb3.9) test  →  clamp [4.0, 10.0]
    bg_offset = min(5.0, n_pass / 3.0)
    std_7b = 3.75 * bkg.temp7_bkg_stddev + bg_offset
    std_7b = max(4.0, min(10.0, std_7b))

    # Std Dev (Reflb) test  →  clamp [0.25, 1.0]
    std_reflb = 3.0 * bkg.rad_diff_sigma
    std_reflb = max(0.25, min(1.0, std_reflb))

    # Std Dev (Reflb) max test  →  clamp [2.5, 10.0]
    std_reflb_max = 2.5 * bkg.rad_diff_sigma + 0.5 * min(5.0, n_pass / 3.0)
    std_reflb_max = max(2.5, min(10.0, std_reflb_max))

    return std_7b14b, std_7b, std_reflb, std_reflb_max


# ── TPW look-up correction (ATBD 3.4.2.8) ────────────────────────────────────
def _tpw_lut_indices(tpw_mm: float, lza_deg: float) -> tuple[int, int]:
    """Return (row_offset, col_index) for the LUT lookup."""
    bin_tpw = int(round(tpw_mm / 10.0))
    bin_tpw = max(1, min(5, bin_tpw))
    bin_ang = int(round(lza_deg / 10.0))
    bin_ang = max(1, min(7, bin_ang))
    col = 7 * (bin_tpw - 1) + bin_ang - 1   # 0-based
    return col


def _apply_tpw_correction(rad: float, ext: float, trans: float) -> float:
    """
    radcorr = (rad - ext * rad) / trans
    ATBD: radcorr = (rad - ext * rad_offset) / trans
    ext is the absorption offset (in radiance units), not a fraction.
    """
    return (rad - ext * rad) / trans


# ── Solar reflectivity correction ────────────────────────────────────────────
def _solar_correction(
    rad7_corr_emiss:    float,
    rad7_bkg_corr:      float,
    rad14_bkg_corr:     float,
    emiss7:             float,
) -> float:
    """
    radsolar = rad7_bkg_corr/emiss7 - emiss7 * rad7from14_bkg_corr/emiss7
             = rad7_bkg_corr_emiss - emiss7 * planck_ch7(T_from_rad14_bkg_corr)

    Returns rad7_corr corrected for solar reflectivity.
    """
    # Brightness temperature from background 14 µm corrected radiance
    T_bkg14 = planck_temp(14, rad14_bkg_corr)
    # Convert to Ch7 radiance space (Planck at Ch7 wavelength with T_bkg14)
    rad7from14_bkg = planck_rad(7, T_bkg14)

    # Solar component
    rad7_bkg_corr_emiss = rad7_bkg_corr / emiss7
    rad_solar = rad7_bkg_corr_emiss - emiss7 * rad7from14_bkg

    # Apply solar correction
    rad7_solar_corr = (rad7_corr_emiss - rad_solar) / emiss7
    return rad7_solar_corr, rad7from14_bkg


# ── Main Part I function ──────────────────────────────────────────────────────
def run_part1(
    # ── ABI band data ────────────────────────────────────────────────────────
    bt7:   np.ndarray,
    rad7:  np.ndarray,
    bt14:  np.ndarray,
    rad14: np.ndarray,
    bt13:  Optional[np.ndarray],
    rad13: Optional[np.ndarray],
    bt15:  Optional[np.ndarray],
    refl2: Optional[np.ndarray],       # Channel 2 reflectance
    # ── Geometry ─────────────────────────────────────────────────────────────
    latitudes:   np.ndarray,
    longitudes:  np.ndarray,
    sza:         np.ndarray,           # Solar zenith angle [deg]
    glint_angle: np.ndarray,           # Sun glint angle [deg]
    lza:         np.ndarray,           # Local (satellite) zenith angle [deg]
    azimuth:     np.ndarray,           # Relative azimuth [deg]
    # ── Auxiliary dynamic ────────────────────────────────────────────────────
    tpw:         np.ndarray,           # Total precipitable water [mm]
    emiss7:      np.ndarray,           # Surface emissivity band 7
    emiss14:     np.ndarray,           # Surface emissivity band 14
    lut_tpw:     np.ndarray,           # TPW LUT (6 rows × 35 cols)
    FPT:         float,                # Focal Plane Temperature [K]
    # ── Auxiliary static ─────────────────────────────────────────────────────
    land_cover:  np.ndarray,           # MODIS land mask values
    land_mask:   np.ndarray,           # Binary land mask (True = land)
    desert_mask: np.ndarray,           # Desert mask
    usgs_eco:    np.ndarray,           # USGS ecosystem type
    data_quality: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, List[FireCandidate]]:
    """
    Part I: loop over all pixels, identify fire candidates, apply corrections.

    Returns
    -------
    fire_mask : 2-D int array  (FireMask codes)
    fail_char_arr : 2-D int array (FailChar codes)
    candidates : list of FireCandidate objects for Part II
    """
    L, W = bt7.shape

    # ── Initialize outputs ────────────────────────────────────────────────────
    fire_mask     = np.full((L, W), FireMask.FIRE_FREE, dtype=np.int32)
    fail_char_arr = np.full((L, W), FailChar.NONE,      dtype=np.int32)
    candidates:  List[FireCandidate] = []
    fire_id_ctr  = 0

    # ── Daylight flag ─────────────────────────────────────────────────────────
    is_day = (sza >= 0.0) & (sza <= MAX_SZA_DAYLIGHT)
    sza_cos = np.cos(np.radians(sza))

    # ── Albedo and visible brightness (ATBD 3.4.2.1) ─────────────────────────
    albedo         = np.full((L, W), np.nan)
    vis_brightness = np.zeros ((L, W), dtype=np.float32)
    if refl2 is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            albedo = np.where(is_day, refl2 / np.where(sza_cos > 0, sza_cos, np.nan), np.nan)
        vis_brightness = (255.0 * np.sqrt(np.clip(refl2, 0, None))).astype(np.float32)

    # ── FPT mitigation: build hybrid longwave band (ATBD 3.4.2.2) ────────────
    use_hybrid = FPT > FPT_THRESHOLD
    if use_hybrid and bt13 is not None and rad13 is not None:
        rad13_in_ch7 = planck_rad(7, bt13)   # ch13 BT → ch7 radiance space
        rad14_in_ch7 = planck_rad(7, bt14)   # ch14 BT → ch7 radiance space
        refl_7_14 = rad7 - rad14_in_ch7
        refl_7_13 = rad7 - rad13_in_ch7
        # Pixel-by-pixel: choose smallest absolute radiance difference
        use_ch13 = np.abs(refl_7_13) < np.abs(refl_7_14)
        # Rebuild bt14 and rad14 as hybrid
        bt14_eff  = np.where(use_ch13, bt13,  bt14)
        rad14_eff = np.where(use_ch13, rad13, rad14)
    else:
        bt14_eff  = bt14
        rad14_eff = rad14

    # ── Reflectivity product (Refl) for all pixels (ATBD 3.4.2.2) ────────────
    rad14_in_7 = planck_rad(7, bt14_eff)
    refl       = rad7 - rad14_in_7
    # Where any input radiance < 0 → set Refl = -9999
    bad_rad = (rad7 < 0) | (rad14_eff < 0)
    refl    = np.where(bad_rad, -9999.0, refl)

    # ── Pixel loop ────────────────────────────────────────────────────────────
    # Cache for background (skip recompute for same scan element if prev was OK)
    prev_j_bkg: Optional[int]    = None
    prev_bkg:   Optional[BackgroundStats] = None

    for i in range(L):
        for j in range(W):

            # ── Space pixel ───────────────────────────────────────────────────
            if np.isnan(bt7[i, j]):
                fire_mask[i, j] = FireMask.SPACE
                continue

            # ATBD 3.4.2.3 ─────────────────────────────────────────────────────

            # Satellite zenith angle test
            if lza[i, j] > MAX_LOCAL_ZENITH:
                fire_mask[i, j] = FireMask.ZENITH_BLOCK
                continue

            # Sun-glint / sub-solar block-out
            if lza[i, j] < GLINT_THRESHOLD or glint_angle[i, j] < GLINT_THRESHOLD:
                fire_mask[i, j] = FireMask.GLINT_BLOCK

            # Bad/unusable pixels
            MISS_VAL = np.nan
            if np.isnan(bt7[i, j]):
                fire_mask[i, j] = FireMask.MISS_CH7;   continue
            if np.isnan(bt14_eff[i, j]):
                fire_mask[i, j] = FireMask.MISS_CH14;  continue
            if bt7[i, j] > SAT_TEMP_CH7  + SAT_BUFF:
                fire_mask[i, j] = FireMask.SAT_CH7;    continue
            if bt14_eff[i, j] > SAT_TEMP_CH14 + SAT_BUFF:
                fire_mask[i, j] = FireMask.SAT_CH14;   continue
            if bt7[i, j] < MIN_BT:
                fire_mask[i, j] = FireMask.UNUS_CH7;   continue
            if bt14_eff[i, j] < MIN_BT:
                fire_mask[i, j] = FireMask.UNUS_CH14;  continue

            # ── Ecosystem / surface mask tests ────────────────────────────────
            if (land_cover[i, j] in MODIS_WATER_CODES
                    or land_cover[i, j] == UMD_WATER_CODE
                    or desert_mask[i, j] == DESERT_BRIGHT):
                fire_mask[i, j] = FireMask.BAD_ECOSYSTEM;  continue

            eco = usgs_eco[i, j]
            if eco == USGS_SEA_WATER:
                fire_mask[i, j] = FireMask.SEA_WATER;     continue
            if eco in USGS_COAST_FRINGE:
                fire_mask[i, j] = FireMask.COAST_FRINGE;  continue
            if eco in USGS_INLAND_WATER:
                fire_mask[i, j] = FireMask.INLAND_WATER;  continue

            # ── Radiance quality check ────────────────────────────────────────
            if rad7[i, j] < 0 or rad14_eff[i, j] < 0:
                fire_mask[i, j] = FireMask.NEG_RAD;        continue

            # ── Minimum BT difference (warm pixels) ──────────────────────────
            diff_bt = bt7[i, j] - bt14_eff[i, j]
            if (bt7[i, j] > 273 or bt14_eff[i, j] > 273) and abs(diff_bt) <= 2.0:
                fire_mask[i, j] = FireMask.FIRE_FREE;      continue
            if abs(diff_bt) < 2.0 and (bt7[i, j] <= 273 or bt14_eff[i, j] <= 273):
                fire_mask[i, j] = FireMask.TOO_COLD;       continue

            # ── Day-dependent thresholds ──────────────────────────────────────
            sc = sza_cos[i, j]
            day_pixel = bool(is_day[i, j])
            bt7_min      = BT7_MIN_NIGHT + (BT7_MIN_SOLAR_COEF      * sc if day_pixel else 0)
            bt7_refl_thr = BT7_REFL_THRESH_NIGHT + (BT7_REFL_THRESH_SOLAR * sc if day_pixel else 0)

            # ── Cloud tests (ATBD 3.4.2.3) ────────────────────────────────────
            is_cloudy = False
            if fire_mask[i, j] == FireMask.FIRE_FREE:

                if bt14_eff[i, j] < CLOUD_BT14_THRESH:
                    fire_mask[i, j] = FireMask.CLOUD_BT14;  is_cloudy = True

                if bt7[i, j] - bt14_eff[i, j] < CLOUD_BT7_BT14_NEG:
                    fire_mask[i, j] = FireMask.CLOUD_BT14_BT15_NEG; is_cloudy = True

                if (bt7[i, j] - bt14_eff[i, j] > CLOUD_BT7_BT14_POS
                        and bt7[i, j] < CLOUD_BT7_FOR_POS):
                    fire_mask[i, j] = FireMask.CLOUD_BT7_BT14_POS; is_cloudy = True

                if day_pixel and refl2 is not None:
                    sza_d = sza[i, j]
                    if (sza_d <= 70 or (sza_d <= 60 and lza[i, j] <= 60)):
                        if not np.isnan(albedo[i, j]) and albedo[i, j] > CLOUD_ALBEDO_THRESH:
                            fire_mask[i, j] = FireMask.CLOUD_ALBEDO; is_cloudy = True

                if bt15 is not None:
                    if bt15[i, j] <= CLOUD_BT15_THRESH:
                        fire_mask[i, j] = FireMask.CLOUD_BT15; is_cloudy = True
                    if bt14_eff[i, j] < CLOUD_BT14_THRESH:
                        if bt14_eff[i, j] - bt15[i, j] < CLOUD_BT14_BT15_NEG:
                            fire_mask[i, j] = FireMask.CLOUD_BT14_BT15_NEG; is_cloudy = True
                        if bt14_eff[i, j] - bt15[i, j] > CLOUD_BT14_BT15_POS:
                            fire_mask[i, j] = FireMask.CLOUD_BT14_BT15_POS; is_cloudy = True

            # ── Along-scan reflectivity test (ATBD 3.4.2.4) ──────────────────
            alb_ij  = albedo[i, j] if refl2 is not None else np.nan
            refl_ij = refl[i, j]

            def _refl_neighbor(jj):
                jj = max(0, min(W-1, jj))
                return refl[i, jj]

            if day_pixel and not np.isnan(alb_ij):
                if alb_ij >= CLOUD_ALBEDO_THRESH and bt7[i, j] < 320.0:
                    if _refl_neighbor(j-3) < 0.2 or _refl_neighbor(j+3) < 0.2:
                        fire_mask[i, j] = FireMask.ALONG_SCAN_DAY;   continue
            else:
                if bt7[i, j] < bt7_min and bt7[i, j] >= MIN_BT:
                    if _refl_neighbor(j-3) < 0.2 or _refl_neighbor(j+3) < 0.2:
                        fire_mask[i, j] = FireMask.ALONG_SCAN_NIGHT; continue

            # ── Saturation flag ───────────────────────────────────────────────
            sat_flag = (bt7[i, j] >= SAT_FLAG_CH7 or bt14_eff[i, j] >= SAT_FLAG_CH14)
            if sat_flag:
                fail_char_arr[i, j] = FailChar.F7

            # ── Background statistics (ATBD 3.4.2.5) ─────────────────────────
            # Reuse previous background if same scan element
            if prev_j_bkg == j and prev_bkg is not None:
                bkg = prev_bkg
            else:
                bkg = compute_background(
                    i, j,
                    bt7, bt14_eff, refl,
                    vis_brightness, albedo if refl2 is not None else None,
                    land_mask,
                    sza_cos, is_day,
                )
                prev_j_bkg = j
                prev_bkg   = bkg

            if bkg is None:
                fire_mask[i, j] = FireMask.NO_BACKGROUND
                continue

            n_pass = bkg.n_passes

            # ── Contextual thresholds (ATBD 3.4.2.6) ─────────────────────────
            std_7b14b, std_7b, std_reflb, std_reflb_max = _contextual_thresholds(bkg, sc)

            # Along-scan ±2 test (used in 3.4.2.6)
            def _along_scan_test() -> bool:
                r_m2 = _refl_neighbor(j-2)
                r_p2 = _refl_neighbor(j+2)
                if abs(r_m2) < std_reflb and abs(r_p2) < std_reflb:
                    if bt7[i, j] < bt7_refl_thr:
                        return False
                return True

            pass_along = _along_scan_test()

            # Background albedo (daylight only)
            alb_bkg = np.nan
            if day_pixel and not np.isnan(bkg.vis_mean_bkg):
                vis_cnt = bkg.vis_mean_bkg
                alb_bkg = ((vis_cnt / 25.5) ** 2) / (max(sc, 1e-6) * 100.0)

            # ── Apply thresholds to identify fire pixels (ATBD 3.4.2.7) ──────
            # Saturated / max-iterations quick path
            if sat_flag or n_pass > BKG_MAX_ITER:
                if ((bt7[i, j] - bt14_eff[i, j] < std_7b14b)
                        and (bt7[i, j] - bkg.temp7_bkg_mean < std_7b)):
                    continue   # Not a fire

            fire_size_init = 0.0
            fire_temp_init = 0.0 if sat_flag else -9.05

            # Non-fire tests (no flag recorded)
            if (refl_ij < std_reflb and bt7[i, j] < 320.0
                    and (bt7[i, j] - bt14_eff[i, j] < 0
                         or bt7[i, j] - bkg.temp7_bkg_mean < 0)):
                continue

            # FailChar = 1
            fc = FailChar.NONE
            if ((bt7[i, j] - bt14_eff[i, j] < std_7b14b)
                    and (refl_ij < std_reflb_max or pass_along)):
                fc = FailChar.F1

            # FailChar = 2
            if ((bt7[i, j] - bkg.temp7_bkg_mean < std_7b)
                    and (refl_ij < std_reflb_max or pass_along)):
                if fc == FailChar.NONE:
                    fc = FailChar.F2

            fail_char_arr[i, j] = fc

            # ── Corrections and adjustments (ATBD 3.4.2.8) ───────────────────
            col = _tpw_lut_indices(float(tpw[i, j]), float(lza[i, j]))
            trans7  = float(lut_tpw[2, col])
            trans14 = float(lut_tpw[3, col])
            ext7    = float(lut_tpw[4, col])
            ext14   = float(lut_tpw[5, col])

            r7  = float(rad7     [i, j])
            r14 = float(rad14_eff[i, j])

            # TPW correction
            try:
                r7_corr  = (r7  - ext7  * r7 ) / trans7
                r14_corr = (r14 - ext14 * r14) / trans14
            except ZeroDivisionError:
                fire_mask[i, j] = FireMask.CONV_ERROR;  continue

            if r7_corr < 0 or r14_corr < 0:
                fire_mask[i, j] = FireMask.CONV_ERROR;  continue

            T7_corr  = float(planck_temp(7,  r7_corr))
            T14_corr = float(planck_temp(14, r14_corr))
            if T7_corr <= 0 or T14_corr <= 0:
                fire_mask[i, j] = FireMask.CONV_ERROR;  continue

            # Smoke/thin cloud correction (daylight, albedo diff in (0.025, 0.07))
            if day_pixel and refl2 is not None and not np.isnan(alb_ij) and not np.isnan(alb_bkg):
                alb_diff = alb_ij - alb_bkg
                if CLOUD_ADJ_ALBEDO_LOW < alb_diff < CLOUD_ADJ_ALBEDO_HIGH:
                    T7_corr  += CLOUD_ADJ_BT7_COEF  * alb_diff
                    T14_corr += CLOUD_ADJ_BT14_COEF * alb_diff
                    fail_char_arr[i, j] = FailChar.F8
                elif alb_ij > CLOUD_ALBEDO_THRESH or alb_diff >= CLOUD_ADJ_ALBEDO_HIGH:
                    T7_corr  += CLOUD_ADJ_BT7_FIXED
                    T14_corr += CLOUD_ADJ_BT14_FIXED
                    fail_char_arr[i, j] = FailChar.F8

            # Emissivity correction
            em7  = float(emiss7 [i, j])
            em14 = float(emiss14[i, j])
            r7_corr_em  = r7_corr  / em7
            r14_corr_em = r14_corr / em14

            # Background radiances (must be converted from BT → rad BEFORE TPW correction)
            r7_bkg_raw  = planck_rad(7,  bkg.temp7_bkg_mean)
            r14_bkg_raw = planck_rad(14, bkg.temp14_bkg_mean)
            r7_bkg_corr  = (r7_bkg_raw  - ext7  * r7_bkg_raw ) / trans7
            r14_bkg_corr = (r14_bkg_raw - ext14 * r14_bkg_raw) / trans14

            if r7_bkg_corr <= 0 or r14_bkg_corr <= 0:
                fire_mask[i, j] = FireMask.CONV_ERROR;  continue

            # Solar reflectivity correction
            try:
                r7_solar_corr, rad7from14_bkg = _solar_correction(
                    r7_corr_em, r7_bkg_corr, r14_bkg_corr, em7
                )
            except Exception:
                fire_mask[i, j] = FireMask.CONV_ERROR;  continue

            if r7_solar_corr <= 0:
                fire_mask[i, j] = FireMask.CONV_ERROR;  continue

            # Diffraction correction
            r7_diff  = (r7_solar_corr - DIFFRAC_CH7_SUB  * rad7from14_bkg) / DIFFRAC_CH7_DIV
            r14_diff = (r14_corr      - DIFFRAC_CH14_SUB * r14_bkg_corr)   / DIFFRAC_CH14_DIV

            if r7_diff <= 0 or r14_diff <= 0:
                fire_mask[i, j] = FireMask.CONV_ERROR;  continue

            T7c  = float(planck_temp(7,  r7_diff))
            T14c = float(planck_temp(14, r14_diff))
            Tbc  = float(planck_temp(7,  r7_bkg_corr))   # corrected background

            if T7c <= 0 or T14c <= 0 or Tbc <= 0:
                fire_mask[i, j] = FireMask.CONV_ERROR;  continue

            # ── Post-correction tests (ATBD 3.4.2.9) ──────────────────────────
            fc = int(fail_char_arr[i, j])
            offset_day = (BT7_MIN_SOLAR_COEF * sc) if day_pixel else 0.0

            if T14c < 285.0 or T7c < (285.0 + offset_day):
                fc = FailChar.F3

            elif T14c - Tbc < 0.25:
                if ((not np.isnan(alb_ij) and alb_ij > 0.15) or is_cloudy) \
                        and T7c - Tbc > 10.0:
                    fc = FailChar.F10
                else:
                    fc = FailChar.F4

            elif T7c - Tbc < 2.0:
                fc = FailChar.F5

            # Sun-glint flag
            if day_pixel and not np.isnan(alb_ij):
                if alb_ij >= 0.25 or (not np.isnan(alb_bkg) and alb_ij - alb_bkg > 0.07):
                    fc = FailChar.F8

            fail_char_arr[i, j] = fc

            # Pixels with fc ∈ {3,4,5,10} skip Dozier, go to last-chance
            skip_dozier = fc in (FailChar.F3, FailChar.F4, FailChar.F5, FailChar.F10)

            # ── Dozier sub-pixel characterization (ATBD 3.4.2.10) ─────────────
            doz = DozierResult()
            if not skip_dozier and not sat_flag:
                is_glint = (fc == FailChar.F8)
                doz = compute_dozier(
                    r7_diff, r14_diff,
                    r7_bkg_corr, r14_bkg_corr,
                    Tbc,
                    is_potential_glint=is_glint,
                )
                # Update fail_char from Dozier result
                if doz.fail_char != FailChar.NONE:
                    fail_char_arr[i, j] = doz.fail_char
                    fc = doz.fail_char
                elif doz.valid:
                    fail_char_arr[i, j] = FailChar.NONE

            # ── Pixel area (ATBD 3.4.2.10) ────────────────────────────────────
            pix_area = compute_pixel_area(i, j, latitudes, longitudes)

            # ── Last-chance fire test (ATBD 3.4.2.11) ────────────────────────
            if skip_dozier or not doz.valid:
                lct1 = ((bt7[i, j] - bkg.temp7_bkg_mean >= std_7b)
                        and (bt14_eff[i, j] - bkg.temp14_bkg_mean >= -20.0))
                lct2 = ((refl_ij - bkg.reflb >= std_reflb_max) and pass_along)
                if lct1 or lct2:
                    doz.fire_frac = 0.0
                    doz.fire_area = 0.0
                    Tt = doz.fire_temp if not np.isnan(doz.fire_temp) else 0.0
                    if MAX_SURF_TEMP < Tt <= MIN_FIRE_TEMP:
                        doz.fire_temp = -abs(Tt)
                    else:
                        doz.fire_temp = -999.0
                    doz.valid = True   # "last-chance" fire
                else:
                    continue  # Not a fire

            # Recalculate pixel area if needed
            if pix_area < MIN_PIXEL_AREA or pix_area < 0:
                pix_area = compute_pixel_area(i, j, latitudes, longitudes)
            if pix_area < 0:
                fire_mask[i, j] = 188;  continue

            doz.fire_area = float(doz.fire_frac or 0.0) * pix_area

            # ── FRP (ATBD 3.4.2.12) ────────────────────────────────────────────
            fc_now = int(fail_char_arr[i, j])
            # No FRP for saturated (11), cloud (12), low prob (15) or max passes
            if fc_now not in (FailChar.F7,) and n_pass <= BKG_MAX_ITER:
                frp_val = compute_frp(pix_area, float(r7_diff), float(r7_bkg_corr))
            else:
                frp_val = -9.0

            # ── Record candidate (ATBD 3.4.2.13) ──────────────────────────────
            fire_id_ctr += 1
            cand = FireCandidate(
                i=i, j=j,
                lat=float(latitudes [i, j]),
                lon=float(longitudes[i, j]),
                fire_id=fire_id_ctr,
                bt7=float(bt7[i, j]),
                bt14=float(bt14_eff[i, j]),
                rad7=r7, rad14=r14,
                bt7_corr=T7c, bt14_corr=T14c, bt_bkg_corr=Tbc,
                bt7_bkg=bkg.temp7_bkg_mean,
                bt14_bkg=bkg.temp14_bkg_mean,
                bt7_bkg_std=bkg.temp7_bkg_stddev,
                bt14_bkg_std=bkg.temp14_bkg_stddev,
                reflb=bkg.reflb,
                rad_diff_sigma=bkg.rad_diff_sigma,
                n_passes=n_pass,
                std_dev_7b_14b=std_7b14b,
                std_dev_7b=std_7b,
                std_dev_reflb=std_reflb,
                std_dev_reflb_max=std_reflb_max,
                fire_temp=doz.fire_temp,
                fire_frac=float(doz.fire_frac or 0.0),
                fire_area=float(doz.fire_area or 0.0),
                frp=frp_val,
                fail_char=int(fail_char_arr[i, j]),
                is_saturated=sat_flag,
                is_cloudy=is_cloudy,
                is_day=day_pixel,
                sza=float(sza[i, j]),
                lza=float(lza[i, j]),
                azimuth=float(azimuth[i, j]),
                albedo=float(alb_ij) if not np.isnan(alb_ij) else np.nan,
                albedo_bkg=float(alb_bkg) if not np.isnan(alb_bkg) else np.nan,
                vis_count=float(vis_brightness[i, j]),
                vis_bkg=float(bkg.vis_mean_bkg),
                emiss7=em7, emiss14=em14,
                refl_pixel=refl_ij,
                refl_m2=float(_refl_neighbor(j-2)),
                refl_p2=float(_refl_neighbor(j+2)),
                pass_along_scan=pass_along,
                bkg=bkg,
            )
            candidates.append(cand)

    return fire_mask, fail_char_arr, candidates


# Expose BKG_MAX_ITER for part2
from .constants import BKG_MAX_ITER
