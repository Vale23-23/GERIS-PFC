"""
Background condition statistics (ATBD Section 3.4.2.5)

Computes the expanding-window background statistics for a given pixel (i, j).
Returns a dataclass with all 26 quantities listed in Table 3.6.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .constants import (
    BKG_WINDOW_INIT, BKG_WINDOW_STEP, BKG_MAX_ITER, BKG_VALID_FRAC,
    BKG_WARM_BT7_NIGHT, BKG_WARM_SOLAR_COEF, BKG_COLD_THRESH,
    BKG_MIN_VIS, BKG_MAX_ALBEDO, FireMask,
)


@dataclass
class BackgroundStats:
    """All 26 background quantities from Table 3.6."""
    # ── Primary chosen values (used in threshold tests) ──────────────────────
    temp7_bkg_mean:  float = np.nan   # Chosen mean T_ch7  [K]
    temp14_bkg_mean: float = np.nan   # Chosen mean T_ch14 [K]
    vis_mean_bkg:    float = np.nan   # Chosen mean visible brightness [8-bit]

    temp7_bkg_stddev:  float = np.nan
    temp14_bkg_stddev: float = np.nan
    vis_bkg_histogram_stddev: float = np.nan
    histogram_bin_largest_count: float = np.nan

    n_passes:       int   = 0         # Number of window expansions
    n_valid:        int   = 0         # Number of valid background pixels
    bkg_count_frac: float = np.nan    # Fraction of window that was valid

    sum_temp7:  float = np.nan
    sum_temp14: float = np.nan

    # ── Histogram-based values ────────────────────────────────────────────────
    temp7_bkg_histogram:         float = np.nan
    temp14_bkg_histogram:        float = np.nan
    temp7_bkg_histogram_stddev:  float = np.nan
    temp14_bkg_histogram_stddev: float = np.nan
    std_dev_7_14_diff:           float = np.nan

    vis_diff_histogram:      float = np.nan
    vis_histogram_variance:  float = np.nan
    vis_histogram_stddev:    float = np.nan

    # ── Full-window (non-filtered) averages ───────────────────────────────────
    temp7_bkg_avg:  float = np.nan
    temp7_stddev:   float = np.nan
    temp14_bkg_avg: float = np.nan
    temp14_stddev:  float = np.nan

    # ── Reflectivity product background ──────────────────────────────────────
    reflb:          float = np.nan    # Rad_4mu_11mu_avg_diff (mean Refl in window)
    rad_diff_sigma: float = np.nan    # std dev of Refl in window

    # ── Window geometry ───────────────────────────────────────────────────────
    half_width:     int   = 0         # Final half-width of the window


def _is_warm(bt7: float, sza_cos: float, is_day: bool) -> bool:
    """True if pixel is anomalously warm (excluded from background)."""
    thresh = BKG_WARM_BT7_NIGHT
    if is_day:
        thresh += BKG_WARM_SOLAR_COEF * sza_cos
    return bt7 > thresh


def _valid_background_mask(
    i0: int, j0: int,
    half: int,
    bt7: np.ndarray,
    bt14: np.ndarray,
    vis: np.ndarray,
    albedo: Optional[np.ndarray],
    land_mask: np.ndarray,
    sza_cos: np.ndarray,
    is_day: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build boolean mask of valid background pixels in the window centred at (i0, j0).

    Returns
    -------
    valid_mask  : 2-D bool array (shape of full image), True where pixel is valid
    in_window   : 2-D bool array, True inside the window (including out-of-bounds)
    window_size : total number of positions in the (2*half+1)^2 window
                  (out-of-bounds count toward denominator per ATBD)
    """
    L, W = bt7.shape
    i_min = i0 - half;  i_max = i0 + half
    j_min = j0 - half;  j_max = j0 + half

    # Window size includes out-of-bounds (per ATBD).
    window_size = (2 * half + 1) ** 2

    # Clip only the data access.  The returned mask is local to the clipped
    # window: allocating a full-image mask for every pixel was the dominant
    # cost of Part I and was unnecessary because the caller immediately sliced
    # it back to these bounds.
    i_lo = max(0, i_min);  i_hi = min(L - 1, i_max)
    j_lo = max(0, j_min);  j_hi = min(W - 1, j_max)
    sl = np.s_[i_lo:i_hi + 1, j_lo:j_hi + 1]

    bt7_win = bt7[sl]
    bt14_win = bt14[sl]
    day_win = is_day[sl]

    # Vectorized equivalents of the scalar screening rules above.
    warm_threshold = np.where(
        day_win,
        BKG_WARM_BT7_NIGHT + BKG_WARM_SOLAR_COEF * sza_cos[sl],
        BKG_WARM_BT7_NIGHT,
    )
    warm = bt7_win > warm_threshold
    cold = (bt7_win < BKG_COLD_THRESH) | (bt14_win < BKG_COLD_THRESH)
    valid_mask = land_mask[sl] & ~warm & ~cold

    # Exclude central pixel 
    center_local_i = i0 - i_lo
    center_local_j = j0 - j_lo
    valid_mask[center_local_i, center_local_j] = False

    # Channel 2/albedo screening applies only when both arrays are available,
    # and only for finite values, matching the previous scalar implementation.
    if day_win is not None and vis is not None and albedo is not None:
        vis_win = vis[sl]
        alb_win = albedo[sl]
        finite_visible = ~np.isnan(vis_win) & ~np.isnan(alb_win)
        reflective = day_win & finite_visible & (
            (vis_win < BKG_MIN_VIS) | (alb_win > BKG_MAX_ALBEDO)
        )
        valid_mask &= ~reflective

    return valid_mask, window_size


def _histogram_select(diff_vals: np.ndarray) -> tuple[np.ndarray, float]:
    """
    ATBD 3.4.2.5 "histogram approach":
    Build ONE histogram from the integer-rounded (T7 - T14) background
    difference values. Find the bin with the highest frequency, along with
    its two closest neighbours (peak-1, peak, peak+1). Returns a boolean
    mask (over diff_vals) selecting the pixels that fall in that 3-bin
    range, to be reused for T7, T14 and visible-brightness statistics.
 
    Returns
    -------
    mask     : boolean array, same length as diff_vals
    peak_val : the integer value of the peak (highest-frequency) bin

    """

    n = len(diff_vals)
    if n == 0:
        return np.zeros(0, dtype=bool), np.nan
 
    rounded = np.round(diff_vals).astype(int)
    bins = np.bincount(rounded - rounded.min())
    peak_idx = np.argmax(bins)
    peak_val = rounded.min() + peak_idx
 
    mask = (rounded >= peak_val - 1) & (rounded <= peak_val + 1)

    return mask, float(peak_val)

def _mean_std(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def compute_background(
    i0: int,
    j0: int,
    bt7: np.ndarray,
    bt14: np.ndarray,
    refl: np.ndarray,
    vis: np.ndarray,
    albedo: Optional[np.ndarray],
    land_mask: np.ndarray,
    sza_cos: np.ndarray,
    is_day: np.ndarray,
    trace: Optional[list] = None,   #If a list is given, its filed with diagnosis foreach iteration
) -> Optional[BackgroundStats]:
    """
    Compute background statistics for pixel (i0, j0).

    Expands the window up to BKG_MAX_ITER times until at least BKG_VALID_FRAC
    of the window is valid cloud/fire-free land pixels.
    Out-of-bounds pixels count toward window denominator but never as valid.

    Returns None if maximum iterations reached without satisfying the criterion
    (caller should set FireMask.NO_BACKGROUND).
    """
    L, W = bt7.shape
    half   = BKG_WINDOW_INIT
    n_iter = 0
    stats  = None

    while n_iter <= BKG_MAX_ITER:
        valid_mask, window_size = _valid_background_mask(
            i0, j0, half, bt7, bt14, vis, albedo, land_mask, sza_cos, is_day
        )

        # Clip to actual image area for data extraction
        i_lo = max(0, i0 - half);  i_hi = min(L - 1, i0 + half)
        j_lo = max(0, j0 - half);  j_hi = min(W - 1, j0 + half)

        n_valid = int(valid_mask.sum())

        if trace is not None:
            trace.append({
                'n_iter': n_iter,
                'half': half,
                'window_size': window_size,
                'n_valid': n_valid,
                'frac_valid': n_valid / window_size,
                'cumple_20pct': n_valid >= BKG_VALID_FRAC * window_size,
            })
        if n_valid >= BKG_VALID_FRAC * window_size:
            # ── Extract valid pixel values ──────────────────────────────────
            vm = valid_mask
            bt7_win  = bt7 [i_lo:i_hi+1, j_lo:j_hi+1]
            bt14_win = bt14[i_lo:i_hi+1, j_lo:j_hi+1]
            refl_win = refl[i_lo:i_hi+1, j_lo:j_hi+1]
            vis_win  = vis [i_lo:i_hi+1, j_lo:j_hi+1] if vis is not None else None

            bt7_vals  = bt7_win [vm]
            bt14_vals = bt14_win[vm]
            refl_vals = refl_win[vm]
            vis_vals  = vis_win [vm] if vis_win is not None else np.zeros(n_valid)

            # ── Difference array ────────────────────────────────────────────
            diff_vals = bt7_vals - bt14_vals

            # ── Statistical approach ────────────────────────────────────────
            t7_stat_mean   = float(np.mean(bt7_vals))
            t7_stat_std    = float(np.std (bt7_vals))
            t14_stat_mean  = float(np.mean(bt14_vals))
            t14_stat_std   = float(np.std (bt14_vals))

            # ── Histogram approach (ATBD 3.4.2.5) ────────────────────────────
            # ONE histogram, built on the rounded (T7 - T14) difference. The
            # peak bin (+/- 1 neighbour) selects a subset of pixels that is
            # then reused to compute T7, T14 and visible-brightness means/std
    
            hist_mask, diff_peak = _histogram_select(diff_vals)
            t7_hist_mean,  t7_hist_std  = _mean_std(bt7_vals [hist_mask])
            t14_hist_mean, t14_hist_std = _mean_std(bt14_vals[hist_mask])
            vis_hist_mean, vis_hist_std = _mean_std(vis_vals [hist_mask])
            
            if trace is not None:
                trace.append({
                    'fase': 'enfoques',
                    't7_stat_mean': t7_stat_mean, 't7_stat_std': t7_stat_std,
                    't14_stat_mean': t14_stat_mean, 't14_stat_std': t14_stat_std,
                    't7_hist_mean': t7_hist_mean, 't7_hist_std': t7_hist_std,
                    't14_hist_mean': t14_hist_mean, 't14_hist_std': t14_hist_std,
                    'diff_peak': diff_peak, 'n_hist_selected': int(hist_mask.sum()),
                    'bt7_vals': bt7_vals, 'bt14_vals': bt14_vals,   # arrays, para graficar despues
                    'valid_bounds': (i_lo, i_hi, j_lo, j_hi),
                    'valid_mask': vm,
                })

            # ── Choose approach with lower BT7 std dev (ATBD 3.4.2.5) ────────
            # This single comparison governs both temp7_bkg_mean and
            # temp14_bkg_mean. Vis brightness is a special case: "for daylit
            # pixels, the Channel 2 approach is always based on a histogram"
            # (ATBD), so vis_mean_bkg always comes from the histogram
            # selection regardless of which approach won for T7/T14.

            if t7_stat_std <= t7_hist_std:
                chosen = "stat"
                temp7_mean  = t7_stat_mean
                temp14_mean = t14_stat_mean
        
            else:
                chosen = "hist"
                temp7_mean  = t7_hist_mean
                temp14_mean = t14_hist_mean
            
            vis_mean_bkg = vis_hist_mean

            # ── Full window (non-filtered) stats for debugging ──────────────
            bt7_full  = bt7_win.ravel()
            bt14_full = bt14_win.ravel()

            # ── Reflb and Rad_Diff_Sigma (ATBD Table 3.6) ────────────────────
            # "The mean/std of the Ch7 minus Ch14 radiance difference (in Ch7
            # space) for all [valid] pixels within the immediate vicinity."
            # Must be computed on valid pixels only (not full window).
            refl_valid = refl_vals[refl_vals > -9998.0]
            if refl_valid.size == 0:
                refl_valid = refl_vals  # fallback: all valid were sentinel

            stats = BackgroundStats(
                temp7_bkg_mean           = temp7_mean,
                temp14_bkg_mean          = temp14_mean,
                vis_mean_bkg             = float(vis_mean_bkg) if not np.isnan(vis_mean_bkg) else np.nan,
                temp7_bkg_stddev         = t7_stat_std if chosen == "stat" else t7_hist_std,
                temp14_bkg_stddev        = t14_stat_std if chosen == "stat" else t14_hist_std,
                vis_bkg_histogram_stddev = float(vis_hist_std),
                # Table 3.6: same quantity as vis_diff_histogram (mean vis
                # brightness via the histogram technique); kept as a separate
                # legacy field for output parity with the reference code.
                histogram_bin_largest_count = float(vis_hist_mean),
                n_passes                 = n_iter,
                n_valid                  = n_valid,
                bkg_count_frac           = n_valid / window_size,
                sum_temp7                = float(np.sum(bt7_vals)),
                sum_temp14               = float(np.sum(bt14_vals)),
                temp7_bkg_histogram      = float(t7_hist_mean),
                temp14_bkg_histogram     = float(t14_hist_mean),
                temp7_bkg_histogram_stddev  = float(t7_hist_std),
                temp14_bkg_histogram_stddev = float(t14_hist_std),
                std_dev_7_14_diff        = float(np.std(diff_vals)),
                vis_diff_histogram       = float(vis_hist_mean),
                vis_histogram_variance   = float(vis_hist_std**2),
                vis_histogram_stddev     = float(vis_hist_std),
                temp7_bkg_avg            = float(np.mean(bt7_full)),
                temp7_stddev             = float(np.std (bt7_full)),
                temp14_bkg_avg           = float(np.mean(bt14_full)),
                temp14_stddev            = float(np.std (bt14_full)),
                reflb                    = float(np.mean(refl_valid)),
                rad_diff_sigma           = float(np.std (refl_valid)),
                half_width               = half,
            )
            return stats

        # Not enough valid pixels → expand window
        n_iter += 1
        half   += BKG_WINDOW_STEP

    return None   # Max iterations reached
