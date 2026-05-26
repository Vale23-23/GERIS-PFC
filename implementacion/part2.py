"""
FDCA Part II – Loop over fire candidates
Implements ATBD sections 3.4.2.14 through 3.4.2.18
"""

import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta

from .constants import (
    FireMask, FailChar,
    BKG_MAX_ITER, MIN_FIRE_TEMP, MAX_SURF_TEMP,
    TEMPORAL_WINDOW_H, TEMPORAL_PIXEL_RAD,
)
from .part1 import FireCandidate


def _std_reflb_part2(rad_diff_sigma: float) -> float:
    """
    Std Dev (Reflb) Part II test:
    2.5 * sigma, floor at 2.5 (ATBD 3.4.2.14)
    """
    return max(2.5, 2.5 * rad_diff_sigma)


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
    bg_off = min(5.0, n_pass / 3.0)

    base1 = 7.0 if confidence == "high" else 5.0
    base2 = 7.0 if confidence == "high" else 5.0
    add_c = 5.0 if confidence == "high" else 3.0

    # Threshold 1
    scaled1 = add_c + bg_off + 2.0 * cand.bt7_bkg_std
    thresh1 = max(base1, scaled1)

    # Threshold 2
    diff_std = cand.bkg.std_dev_7_14_diff if cand.bkg else 0.0
    t7_t14_diff = cand.bt7_bkg - cand.bt14_bkg
    scaled2 = add_c + bg_off + t7_t14_diff + 2.0 * diff_std
    thresh2 = max(base2, scaled2)

    return thresh1, thresh2


def _assign_fire_category(cand: FireCandidate, fire_mask: np.ndarray) -> int:
    """
    Determine final fire mask code (10-15) for a validated candidate.
    ATBD 3.4.2.15
    """
    fc = cand.fail_char
    Tt = cand.fire_temp

    # Processed fire: Tt > 400 K and solution is valid
    if Tt is not None and not np.isnan(Tt) and Tt > MIN_FIRE_TEMP:
        return FireMask.PROCESSED

    # Saturated fire: fire_temp was initialised to 0 (no solution attempted)
    if cand.is_saturated and (Tt == 0 or Tt is None or np.isnan(Tt)):
        return FireMask.SATURATED

    # Cloudy / smoke fire
    if fc in (FailChar.F9, FailChar.F10):
        return FireMask.CLOUD_CONTAM

    # Possible fire categories (Tt < 0 used as flag)
    if Tt is not None and not np.isnan(Tt) and Tt < 0:
        return FireMask.LOW_PROB   # default, refined below

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
                      (None if not available → temporal filtering disabled)
    current_epoch   : seconds since 2001-01-01 00:00:00 of current scan

    Returns
    -------
    fire_mask, fail_char_arr (updated), confirmed_fires list
    """
    TEMPORAL_SECS = TEMPORAL_WINDOW_H * 3600.0
    confirmed: List[FireCandidate] = []

    for cand in candidates:
        i, j = cand.i, cand.j
        bt7   = cand.bt7
        bt7_bkg = cand.bt7_bkg
        refl  = cand.refl_pixel
        reflb = cand.reflb
        sza_cos = np.cos(np.radians(cand.sza))
        is_day = cand.is_day

        std_reflb_p2 = _std_reflb_part2(cand.rad_diff_sigma)

        # ── Threshold tests to eliminate false alarms (ATBD 3.4.2.14) ────────

        # Test 1: BT7 - Tb3.9 < 2 AND (Refl check OR along-scan)
        cond1 = (bt7 - bt7_bkg < 2.0) and \
                (refl - reflb < std_reflb_p2 or _refl_along_scan_part2(cand))
        if cond1:
            continue   # Not a fire

        # Test 2: BT7 too low AND BT7-Tb3.9 < 10 AND BT7-BT14 < 25
        day_off = sza_cos * 20.0 if is_day else 0.0
        cond2 = (bt7 < 290.0 + day_off
                 and bt7 - bt7_bkg < 10.0
                 and bt7 - cand.bt14 < 25.0
                 and (refl - reflb < std_reflb_p2 or _refl_along_scan_part2(cand)))
        if cond2:
            continue

        # Test 3: BT7 low AND Tb3.9 low AND many background passes
        cond3 = (bt7 < 290.0 + day_off
                 and bt7_bkg < 280.0 + day_off
                 and cand.n_passes >= BKG_MAX_ITER
                 and (refl - reflb < std_reflb_p2 or _refl_along_scan_part2(cand)))
        if cond3:
            continue

        # ── Sun-glint re-evaluation ────────────────────────────────────────────
        # ── Cloud/fog edge re-evaluation (ATBD 3.4.2.14) ─────────────────────
        fc = cand.fail_char
        alb  = cand.albedo     if not np.isnan(cand.albedo)     else 0.0
        albb = cand.albedo_bkg if not np.isnan(cand.albedo_bkg) else 0.0

        if fc in (FailChar.F9, FailChar.F10):
            cloud_test = (alb > 0.25 or alb - albb >= 0.10)
            bt_test    = cand.bt7_corr < 292.5 + day_off
            if cloud_test and bt_test:
                cand.fail_char = FailChar.F11
                fail_char_arr[i, j] = FailChar.F11
                fc = FailChar.F11

        # ── Fog/cloud-edge fog flag (ATBD 3.4.2.15 edge case) ─────────────────
        if fc >= FailChar.F9:
            bt7c   = cand.bt7_corr
            Tb7    = cand.bt7_bkg
            Tb14   = cand.bt14_bkg
            bkg_diff = Tb7 - Tb14
            fog_c1 = sza_cos * 20.0 + 5.0 - bkg_diff < 1.5
            fog_c2 = bt7c - Tb7 <= 4.0
            if fog_c1 and fog_c2:
                cand.fail_char = FailChar.F11
                fail_char_arr[i, j] = FailChar.F11
                fc = FailChar.F11

        # ── High / medium probability categorization ──────────────────────────
        fire_code = _assign_fire_category(cand, fire_mask)

        # For possible fire pixels (fc in 3,4,6,8 → attempt up/down-grade)
        Tt = cand.fire_temp
        if fc in (FailChar.F3, FailChar.F4, FailChar.F6, FailChar.F8):
            if Tt is not None and not np.isnan(Tt) and Tt < 0:
                # Check high confidence
                thr1_h, thr2_h = _high_med_thresholds(cand, "high")
                thr1_m, thr2_m = _high_med_thresholds(cand, "medium")

                bt7c   = cand.bt7_corr
                Tb7    = cand.bt7_bkg
                Tb14   = cand.bt14_bkg
                refl_test = (refl - reflb >= cand.std_dev_reflb_max
                             or _refl_along_scan_part2(cand))

                if (bt7c - Tb7 > thr1_h
                        and Tb7 - Tb14 > thr2_h
                        and refl_test):
                    fire_code = FireMask.HIGH_PROB
                    cand.fail_char += 30
                elif (bt7c - Tb7 > thr1_m
                        and Tb7 - Tb14 > thr2_m
                        and refl_test):
                    fire_code = FireMask.MED_PROB
                    cand.fail_char += 20
                else:
                    fire_code = FireMask.LOW_PROB

        # No FRP for passes > 10
        if cand.n_passes > BKG_MAX_ITER:
            cand.frp = -99.0

        # Finalize fire_temp / fire_area for non-processed categories
        if fire_code != FireMask.PROCESSED:
            if Tt is not None and not np.isnan(Tt) and Tt < MIN_FIRE_TEMP:
                cand.fire_area = -999.0
                if fire_code != FireMask.SATURATED:
                    cand.fire_temp = -999.0 if Tt <= MAX_SURF_TEMP else -abs(Tt)

        # ── Temporal filtering (ATBD 3.4.2.16) ───────────────────────────────
        temporally_filtered = False
        if prev_fire_mask is not None:
            L, W = prev_fire_mask.shape
            for di in range(-TEMPORAL_PIXEL_RAD, TEMPORAL_PIXEL_RAD + 1):
                for dj in range(-TEMPORAL_PIXEL_RAD, TEMPORAL_PIXEL_RAD + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < L and 0 <= nj < W:
                        last_t = prev_fire_mask[ni, nj]
                        if last_t > 0 and (current_epoch - last_t) <= TEMPORAL_WINDOW_H * 3600.0:
                            temporally_filtered = True
                            break
                if temporally_filtered:
                    break

        # Apply temporal offset (+20) if filtered
        if temporally_filtered:
            fire_code += 20   # 10→30, 11→31, ... 15→35

        # ── Write final mask code ─────────────────────────────────────────────
        fire_mask[i, j] = fire_code
        confirmed.append(cand)

    return fire_mask, fail_char_arr, confirmed
