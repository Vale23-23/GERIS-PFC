"""
Sub-pixel fire characterization: Dozier method and FRP
Implements ATBD sections 3.4.2.10 and 3.4.2.12
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .planck import (
    planck_rad, planck_temp, planck_deriv_T,   # se mantienen por si algo externo los usa
    planck_rad_from_coeffs, planck_temp_from_coeffs, planck_deriv_T_from_coeffs,
)
from .constants import (
    DOZIER_P_UPPER, DOZIER_P_LOWER, DOZIER_BISECT_N, DOZIER_NEWTON_MAX,
    DOZIER_NEWTON_TOL, MIN_FIRE_TEMP, MAX_SURF_TEMP,
    SIGMA_SB, FRP_MIR_A, LAMBDA,
    FireMask, FailChar,
)


@dataclass
class DozierResult:
    fire_temp:   float = np.nan   # Tt  [K]  (negative = flag, see ATBD)
    fire_frac:   float = np.nan   # p   [0-1]
    fire_area:   float = np.nan   # km²  (set after pixel area calculation)
    frp:         float = -99.0    # Fire Radiative Power [MW]
    fail_char:   int   = FailChar.NONE
    valid:       bool  = False    # True if a fire solution was found


def _dozier_rad_fire(rad_total: float, rad_bkg: float, p: float) -> float:
    """
    Equation A = B + C  →  B/p = (A - (1-p)*C_bkg) / p
    Radiance attributable to fire component per unit fire fraction.
    """
    return (rad_total - (1.0 - p) * rad_bkg) / p


def _solve_bisection(
    rad7: float, rad14: float,
    rad7_bkg: float, rad14_bkg: float,
    coeffs7: dict, coeffs14: dict,
) -> tuple[float, float]:
    """
    15-iteration logarithmic bisection to find fire proportion p.
    ATBD eq. 3.1: p_mid = 10^(log10(p_lo) + (log10(p_hi)-log10(p_lo))/2)

    Returns (p_mid, Tt) after final iteration.
    """
    p_lo = DOZIER_P_LOWER
    p_hi = DOZIER_P_UPPER

    # Sign function for temperature difference at a given p
    def temp_diff_sign(p: float) -> float:
        rf7  = _dozier_rad_fire(rad7,  rad7_bkg,  p)
        rf14 = _dozier_rad_fire(rad14, rad14_bkg, p)
        Tt7  = planck_temp_from_coeffs(rf7,  **coeffs7)
        Tt14 = planck_temp_from_coeffs(rf14, **coeffs14)
        return float(np.sign(Tt7 - Tt14))

    sign_lo = temp_diff_sign(p_lo)

    p_mid = p_lo
    for _ in range(DOZIER_BISECT_N):
        log_mid = np.log10(p_lo) + (np.log10(p_hi) - np.log10(p_lo)) / 2.0
        p_mid   = 10.0 ** log_mid
        sign_mid = temp_diff_sign(p_mid)

        if sign_mid == sign_lo:
            p_lo    = p_mid
            sign_lo = sign_mid
        else:
            p_hi = p_mid

    # Estimate Tt from final p
    rf7 = _dozier_rad_fire(rad7, rad7_bkg, p_mid)
    Tt  = float(planck_temp_from_coeffs(rf7, **coeffs7))
    return p_mid, Tt


def _solve_newton(
    p0: float, Tt0: float,
    rad7: float, rad14: float,
    rad7_bkg: float, rad14_bkg: float,
    Tb: float,
    coeffs7: dict, coeffs14: dict,
) -> tuple[float, float, bool]:
    """
    Newton-Raphson refinement of (p, Tt).
    Solves f = B + C - A ≈ 0 for both channels simultaneously.

    f_k(p, T) = p*L_k(T) + (1-p)*L_k(Tb) - A_k   for k in {7, 14}

    Jacobian:
        J[k,0] = ∂f_k/∂p  = L_k(T) - L_k(Tb)
        J[k,1] = ∂f_k/∂T  = p * dL_k/dT

    Returns (p, Tt, converged)
    """
    p  = p0
    Tt = Tt0

    A7  = rad7
    A14 = rad14
    Lb7  = planck_rad_from_coeffs(Tb, **coeffs7)
    Lb14 = planck_rad_from_coeffs(Tb, **coeffs14)

    converged = False
    for _ in range(DOZIER_NEWTON_MAX):
        if not np.isfinite(Tt) or Tt <= 0:
            break

        B7  = p * planck_rad_from_coeffs(Tt, **coeffs7)
        B14 = p * planck_rad_from_coeffs(Tt, **coeffs14)
        C7  = (1.0 - p) * Lb7
        C14 = (1.0 - p) * Lb14
        f7  = B7  + C7  - A7
        f14 = B14 + C14 - A14
        # Antes: :
        #if abs(f7) < DOZIER_NEWTON_TOL and abs(f14) < DOZIER_NEWTON_TOL:
        if abs(f7) < DOZIER_NEWTON_TOL * abs(A7) and abs(f14) < DOZIER_NEWTON_TOL * abs(A14):
            converged = True
            break

        # Jacobian
        J00 = planck_rad_from_coeffs(Tt, **coeffs7) - Lb7
        J01 = planck_deriv_T_from_coeffs(Tt, p, **coeffs7)
        J10 = planck_rad_from_coeffs(Tt, **coeffs14) - Lb14
        J11 = planck_deriv_T_from_coeffs(Tt, p, **coeffs14)

        det = J00 * J11 - J01 * J10
        if abs(det) < 1e-300:
            break  # Singular Jacobian

        # Inverse of 2×2 matrix times [f7, f14]
        dp  = ( J11 * f7  - J01 * f14) / det
        dTt = (-J10 * f7  + J00 * f14) / det

        p  -= dp
        Tt -= dTt

    return p, Tt, converged


def compute_pixel_area(
    i: int, j: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> float:
    """
    Pixel area in km² using the 4×4 box great-circle method (ATBD 3.4.2.10).

    Corners at (i±2, j±2), distances divided by 4 before area calculation.
    Heron's formula for the parallelogram.
    """
    L, W = latitudes.shape

    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    c = [
        (_clamp(i-2, 0, L-1), _clamp(j-2, 0, W-1)),
        (_clamp(i+2, 0, L-1), _clamp(j-2, 0, W-1)),
        (_clamp(i+2, 0, L-1), _clamp(j+2, 0, W-1)),
        (_clamp(i-2, 0, L-1), _clamp(j+2, 0, W-1)),
    ]

    def great_circle_km(r1, c1, r2, c2):
        lat1 = np.radians(latitudes [r1, c1])
        lat2 = np.radians(latitudes [r2, c2])
        lon1 = np.radians(longitudes[r1, c1])
        lon2 = np.radians(longitudes[r2, c2])
        arg  = (np.sin(lat1)*np.sin(lat2)
                + np.cos(lat1)*np.cos(lat2)*np.cos(abs(lon2-lon1)))
        arg  = np.clip(arg, -1.0, 1.0)
        return 6378.137 * np.arccos(arg)

    # Sides of the 4×4 box divided by 4
    a = great_circle_km(*c[0], *c[1]) / 4.0   # top-left → bottom-left
    b = great_circle_km(*c[1], *c[2]) / 4.0   # bottom-left → bottom-right
    d = great_circle_km(*c[0], *c[2]) / 4.0   # diagonal (used in Heron)

    # Heron for one triangle × 2 → parallelogram area
    s = (a + b + d) / 2.0
    try:
        area = 2.0 * np.sqrt(max(0.0, s * (s-a) * (s-b) * (s-d)))
    except Exception:
        area = 0.0
    return area


def compute_dozier(
    rad7:     float,
    rad14:    float,
    rad7_bkg: float,
    rad14_bkg: float,
    Tb:       float,
    coeffs7:  dict,
    coeffs14: dict,
    is_potential_glint: bool = False,
) -> DozierResult:
    """
    Full Dozier sub-pixel characterization.

    Parameters
    ----------
    rad7, rad14       : fully corrected pixel radiances (W·m⁻²·sr⁻¹·m⁻¹)
    rad7_bkg, rad14_bkg : background radiances
    Tb                : corrected background brightness temperature [K]
    is_potential_glint : True if pixel may be in sun-glint region

    Returns
    -------
    DozierResult with fire_temp, fire_frac (area set later by caller)
    """
    result = DozierResult()

    # ── Bisection ─────────────────────────────────────────────────────────
    try:
        p_bis, Tt_bis = _solve_bisection(
            rad7, rad14, rad7_bkg, rad14_bkg, coeffs7, coeffs14
        )
    except Exception:
        # Bisection failed entirely → last chance
        result.fire_temp = -999.0
        result.fail_char = FailChar.F6
        return result

    # ── Newton refinement ─────────────────────────────────────────────────
    p, Tt, converged = _solve_newton(
        p_bis, Tt_bis,
        rad7, rad14,
        rad7_bkg, rad14_bkg,
        Tb,
        coeffs7, coeffs14,
    ) 

    if Tt <= 0 or not converged:
        # Solution failed → flag for last-chance test
        result.fire_temp = -999.0
        result.fail_char = FailChar.F6
        return result

    # ── Classify result ───────────────────────────────────────────────────
    if Tt < MIN_FIRE_TEMP:
        if is_potential_glint:
            result.fire_temp = Tt
            result.fire_frac = p
            result.fail_char = FailChar.F9   # glint, temp < 400 K
        else:
            result.fire_temp = Tt
            result.fire_frac = p
            result.fail_char = FailChar.F6   # non-glint, temp < 400 K
        return result

    # Valid fire solution (Tt ≥ 400 K)
    if is_potential_glint:
        result.fail_char = FailChar.F9  # overridden to F9 → becomes 9 in Part II
        # Actually if Tt > 400 and was glint→ set to 9 cleared to no-glint processed
    else:
        result.fail_char = FailChar.NONE

    result.fire_temp = Tt
    result.fire_frac = p
    result.valid     = True
    return result


def compute_frp(
    pixel_area: float,
    rad7_obs:   float,
    rad7_bkg:   float,
) -> float:
    """
    Fire Radiative Power via the MIR approximation (ATBD eq. 3.4).

    FRP_MIR = (A_pixel / a) * sigma * (L_MIR - L_B,MIR)   [W] → convert to MW

    Parameters
    ----------
    pixel_area : float  [km²]
    rad7_obs   : float  observed 3.9 µm radiance  [W·m⁻²·sr⁻¹·m⁻¹]
    rad7_bkg   : float  background 3.9 µm radiance [W·m⁻²·sr⁻¹·m⁻¹]

    Returns
    -------
    float [MW]
    """
    area_m2 = pixel_area * 1e6          # km² → m²
    frp_w   = (area_m2 / FRP_MIR_A) * SIGMA_SB * (rad7_obs - rad7_bkg)
    return frp_w / 1e6                  # W → MW