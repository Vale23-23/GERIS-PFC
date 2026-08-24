"""
FDCA Part II - Loop over fire candidates
Implements ATBD sections 3.4.2.14 through 3.4.2.18
"""

import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta

from fdca.constants import *
from .part1 import FireCandidate


def _std_reflb_part2(rad_diff_sigma: float) -> float:
    """
    Std Dev (Reflb) Part II test:
    2.5 * sigma, floor at 2.5 (ATBD 3.4.2.14)
    """
    return max(STD_REFLB_P2_FLOOR, STD_REFLB_P2_SCALE * rad_diff_sigma)

def _eliminate_false_alarm(cand: FireCandidate) -> tuple[bool, str]:
    """
    Threshold tests to eliminate false alarms (ATBD 3.4.2.14).
    If one of the three OR tests is true, the fire candidate is discarded.

    Returns
    -------
    (eliminated, reason) - reason in {"cond1", "cond2", "cond3", ""}
    """
    bt7     = cand.bt7
    bt7_bkg = cand.bt7_bkg
    refl    = cand.refl_pixel
    reflb   = cand.reflb
    sza_cos = np.cos(np.radians(cand.sza))
    is_day  = cand.is_day
    day_off = sza_cos * P2_DAY_OFFSET_COEF if is_day else 0.0

    # calculation of std_reflb_p2 and refl_ok
    std_reflb_p2 = _std_reflb_part2(cand.rad_diff_sigma)
    refl_ok = (refl - reflb < std_reflb_p2) or _refl_along_scan_part2(cand)

    if (bt7 - bt7_bkg < DELTA_BT7_TB7_MIN) and refl_ok:
        return True, "cond1"

    if (bt7 < BT7_WARM_NIGHT + day_off and bt7 - bt7_bkg < DELTA_BT7_TB7_MAX and bt7 - cand.bt14 < DELTA_BT7_BT14_MAX and refl_ok):
        return True, "cond2"

    if (bt7 < BT7_WARM_NIGHT + day_off and bt7_bkg < TB7_COLD_NIGHT + day_off and cand.n_passes >= BKG_MAX_ITER and refl_ok):
        return True, "cond3"

    return False, "" # returns the test that triggered the elimination  

def _reassign_cloud_glint_edge(cand: FireCandidate, day_off: float) -> bool:
    """
    Sun-glint / cloud-fog edge re-evaluation (ATBD 3.4.2.14, second half).

    It only applies to candidates with F9 or F10 flags (see 3.4.2.8-3.4.2.9). 
    If albedo indicates cloud edge/glint and the corrected BT7 is under the threshold,
    it is appropriate to reassign it to F11.

    Returns
    -------
    bool - True if is appropriate to reassign it to F11 (the caller makes the mutation)
    """
    if cand.fail_char not in (FailChar.F9, FailChar.F10):
        return False

    alb  = cand.albedo     if not np.isnan(cand.albedo)     else 0.0
    albb = cand.albedo_bkg if not np.isnan(cand.albedo_bkg) else 0.0

    cloud_test = (alb > CLOUD_EDGE_ALBEDO_MIN or alb - albb >= CLOUD_EDGE_ALBEDO_DIFF_MIN)
    bt_test    = cand.bt7_corr < CLOUD_EDGE_BT7_MAX_NIGHT + day_off
    return cloud_test and bt_test


def _reassign_fog_edge(cand: FireCandidate, sza_cos: float) -> bool:
    """
    Second test to F11 (ATBD 3.4.2.15, condition on BT3.9-BT11.2 of the background)
    Applies to any candidate with flag >=F9.

    """
    if cand.fail_char < FailChar.F9:
        return False

    bkg_diff = cand.bt7_bkg - cand.bt14_bkg
    fog_c1 = sza_cos * P2_DAY_OFFSET_COEF + FOG_EDGE_DAY_OFFSET_ADD - bkg_diff < FOG_EDGE_DIFF_THRESH
    fog_c2 = cand.bt7_corr - cand.bt7_bkg <= FOG_EDGE_BT7_DELTA_MAX
    return fog_c1 and fog_c2


def _upgrade_confidence(cand: FireCandidate) -> tuple[Optional[int], int]:
    """
    High/medium confidence upgrade (ATBD 3.4.2.15).
    It only evaluates candidates with flags in {F3, F4, F6, F8} and fire_temp < 0
    (to whom part I did not find a valid Dozier solution).

    Returns
    -------
    (fire_code, flag_delta): fire_code is None if evaluation is not applicable
    (caller must use _assign_fire_category result in this case).
    flag_delta is 0, 20 or 30, to add to cand.fail_char.
    """
    if cand.fail_char not in (FailChar.F3, FailChar.F4, FailChar.F6, FailChar.F8):
        return None, 0

    Tt = cand.fire_temp
    if Tt is None or np.isnan(Tt) or Tt >= 0:
        return None, 0

    thr1_h, thr2_h = _high_med_thresholds(cand, "high")
    thr1_m, thr2_m = _high_med_thresholds(cand, "medium")

    bt7c = cand.bt7_corr
    Tb7  = cand.bt7_bkg
    Tb14 = cand.bt14_bkg
    refl_test = (cand.refl_pixel - cand.reflb >= cand.std_dev_reflb_max or _refl_along_scan_part2(cand))

    if bt7c - Tb7 > thr1_h and Tb7 - Tb14 > thr2_h and refl_test:
        return FireMask.HIGH_PROB, 30
    if bt7c - Tb7 > thr1_m and Tb7 - Tb14 > thr2_m and refl_test:
        return FireMask.MED_PROB, 20
    return FireMask.LOW_PROB, 0

def _high_med_thresholds(
    cand: FireCandidate, confidence: str
) -> tuple[float, float]:
    """
    High/medium confidence temperature thresholds (ATBD 3.4.2.15).

    Returns (thresh1, thresh2) where:
      thresh1: T3.9 - Tb3.9 must exceed this
      thresh2: Tb3.9 - Tb11.2 must exceed this
    """
    n_pass = cand.n_passes
    bg_off = min(CONF_WINDOW_OFFSET_CAP, n_pass / CONF_WINDOW_OFFSET_DIVISOR)

    base  = CONF_BASE_THRESH_HIGH if confidence == "high" else CONF_BASE_THRESH_MEDIUM
    add_c = CONF_ADD_HIGH         if confidence == "high" else CONF_ADD_MEDIUM

    # Threshold 1
    scaled1 = add_c + bg_off + 2.0 * cand.bt7_bkg_std
    thresh1 = max(base, scaled1)

    # Threshold 2
    diff_std    = cand.bkg.std_dev_7_14_diff if cand.bkg else 0.0
    t7_t14_diff = cand.bt7_bkg - cand.bt14_bkg
    scaled2 = add_c + bg_off + t7_t14_diff + 2.0 * diff_std
    thresh2 = max(base, scaled2)

    return thresh1, thresh2


def _assign_fire_category(cand: FireCandidate, fire_mask: np.ndarray) -> int:
    """
    Determine base fire mask code (10-12, or default 15) for a validated
    candidate, before the high/medium confidence upgrade (_upgrade_confidence)
    is applied by the caller. ATBD 3.4.2.15.
    """
    Tt = cand.fire_temp

    # Processed fire: Tt > 400 K, valid Dozier solution
    if Tt is not None and not np.isnan(Tt) and Tt > MIN_FIRE_TEMP:
        return FireMask.PROCESSED

    # Saturated fire: fire_temp == 0 
    if cand.is_saturated and (Tt == 0 or Tt is None or np.isnan(Tt)):
        return FireMask.SATURATED

    # Cloudy / smoke fire (flag F9 or F10)
    if cand.fail_char in (FailChar.F9, FailChar.F10):
        return FireMask.CLOUD_CONTAM

    # Default: low probability (flag == F11, or < F9, with Tt < 0)
    return FireMask.LOW_PROB


def _refl_along_scan_part2(cand: FireCandidate) -> bool:
    """Along-scan test (same as Part I)."""
    return cand.pass_along_scan

def run_part2(
    candidates:    List[FireCandidate],
    fire_mask:     np.ndarray,
    fail_char_arr: np.ndarray,
    prev_fire_mask: Optional[np.ndarray],   # seconds-since-epoch when last fire
    current_epoch:  float,                  # seconds since 2001-01-01 for current scan
) -> tuple[np.ndarray, np.ndarray, List[FireCandidate]]:
    """
    Part II: further refine the fire product.

    Parameters
    ----------
    candidates      : output of Part I
    fire_mask       : initialized by Part I (will be updated in-place)
    fail_char_arr   : initialized by Part I (will be updated in-place)
    prev_fire_mask  : full-disk array of seconds-since-2001 of last fire at each pixel
                      (None if not available -> temporal filtering disabled)
    current_epoch   : seconds since 2001-01-01 00:00:00 of current scan

    Returns
    -------
    fire_mask, fail_char_arr (updated), confirmed_fires list
    """
    confirmed: List[FireCandidate] = []

    for cand in candidates:
        i, j = cand.i, cand.j
        sza_cos = np.cos(np.radians(cand.sza))
        is_day  = cand.is_day

        # ── 3.4.2.14 (first half): elimination of false alarms ─────────
        eliminated, _reason_314 = _eliminate_false_alarm(cand)
        if eliminated:
            continue
        day_off = sza_cos * P2_DAY_OFFSET_COEF if is_day else 0.0

        # ── 3.4.2.14 (second half): re-evaluation of glint/cloud edge ─────
        fc = cand.fail_char
        if _reassign_cloud_glint_edge(cand, day_off):
            cand.fail_char = FailChar.F11
            fail_char_arr[i, j] = FailChar.F11
            fc = FailChar.F11

        # ── 3.4.2.15: second test of cloud edge/fog ───────────────────
        if _reassign_fog_edge(cand, sza_cos):
            cand.fail_char = FailChar.F11
            fail_char_arr[i, j] = FailChar.F11
            fc = FailChar.F11

        # ── 3.4.2.15: categorization + confidence upgrade ─────────────────
        fire_code = _assign_fire_category(cand, fire_mask)

        upgraded_code, flag_delta = _upgrade_confidence(cand)
        if upgraded_code is not None:
            fire_code = upgraded_code
            cand.fail_char += flag_delta

        # No FRP for passes > 10
        if cand.n_passes > BKG_MAX_ITER:
            cand.frp = FRP_NPASSES_EXCEEDED

        # ── Finalize fire_area for non-processed categories (Table 3.10) ──
        # fire_temp is intentionally NOT touched here: Part I (3.4.2.11)
        # already assigned its final value (-abs(Tt) for smoldering fires in
        # (350K, 400K], or -999.0 otherwise). Table 3.10's "-999 if < 400K"
        # is treated as a summary of that prior step, not a second override

        Tt = cand.fire_temp
        if fire_code != FireMask.PROCESSED:
            if Tt is not None and not np.isnan(Tt) and Tt < MIN_FIRE_TEMP:
                cand.fire_area = LOW_FIRE_SIZE

        # ── Temporal filtering (ATBD 3.4.2.16) ───────────────────────────────
        temporally_filtered = False
        if prev_fire_mask is not None:
            L, W = prev_fire_mask.shape
            for di in range(-TEMPORAL_PIXEL_RAD, TEMPORAL_PIXEL_RAD + 1):
                for dj in range(-TEMPORAL_PIXEL_RAD, TEMPORAL_PIXEL_RAD + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < L and 0 <= nj < W:
                        last_t = prev_fire_mask[ni, nj]
                        if last_t > 0 and (current_epoch - last_t) <= TEMPORAL_WINDOW_S:
                            temporally_filtered = True
                            break
                if temporally_filtered:
                    break

        # Apply temporal offset (+20) if filtered
        if temporally_filtered:
            fire_code += TEMPORAL_FILTER_CODE_OFFSET   # 10->30, 11->31, ... 15->35

        # ── Write final mask code (3.4.2.17/18: output) ──────────────────────
        fire_mask[i, j] = fire_code
        confirmed.append(cand)

    return fire_mask, fail_char_arr, confirmed