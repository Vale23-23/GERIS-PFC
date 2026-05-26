"""
Planck / radiometry utilities for FDCA
All temperatures in Kelvin, radiances in W·m⁻²·sr⁻¹·m⁻¹
"""

import numpy as np
from .constants import H_PLANCK, K_BOLTZ, C_LIGHT, LAMBDA


def planck_rad(band: int, T: np.ndarray) -> np.ndarray:
    """
    Spectral radiance from Planck's law for a given ABI band and temperature.

    Parameters
    ----------
    band : int
        ABI band number (2, 7, 13, 14, 15)
    T    : array-like [K]
        Brightness temperature

    Returns
    -------
    ndarray [W·m⁻²·sr⁻¹·m⁻¹]
    """
    T = np.asarray(T, dtype=np.float64)
    lam = LAMBDA[band]
    c1 = 2.0 * H_PLANCK * C_LIGHT**2
    c2 = H_PLANCK * C_LIGHT / (K_BOLTZ * lam)
    # Guard against T == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        rad = c1 / (lam**5 * (np.exp(c2 / np.where(T > 0, T, np.nan)) - 1.0))
    return rad


def planck_temp(band: int, rad: np.ndarray) -> np.ndarray:
    """
    Brightness temperature from radiance using the inverse Planck function.

    Parameters
    ----------
    band : int
        ABI band number
    rad  : array-like [W·m⁻²·sr⁻¹·m⁻¹]

    Returns
    -------
    ndarray [K]
    """
    rad = np.asarray(rad, dtype=np.float64)
    lam = LAMBDA[band]
    c1 = 2.0 * H_PLANCK * C_LIGHT**2
    c2 = H_PLANCK * C_LIGHT / K_BOLTZ
    with np.errstate(invalid="ignore", divide="ignore"):
        T = (c2 / lam) / np.log1p(c1 / (np.where(rad > 0, rad, np.nan) * lam**5))
    return T


def temp_to_rad_in_band(src_band: int, dst_band: int, T: np.ndarray) -> np.ndarray:
    """
    Convert a brightness temperature from src_band into the equivalent
    radiance expressed in dst_band space.

    This is used to build the 'Reflectivity Product' (Refl):
        Refl = rad_ch7 - rad_ch14_in_ch7_space

    Parameters
    ----------
    src_band : int   Band from which T was measured
    dst_band : int   Target band (radiance space)
    T        : ndarray [K]

    Returns
    -------
    ndarray [W·m⁻²·sr⁻¹·m⁻¹]
    """
    # Step 1: compute radiance in the source band space using Planck
    rad_src = planck_rad(src_band, T)
    # Step 2: that radiance IS the Planck emission at T;
    #         to express in dst_band space we evaluate Planck at dst_band
    #         with the same temperature T (brightness temperature is band-independent
    #         for a blackbody, so we just evaluate Planck at dst_band wavelength).
    rad_dst = planck_rad(dst_band, T)
    return rad_dst


def rad_to_temp_in_band(band: int, rad: np.ndarray) -> np.ndarray:
    """Thin wrapper kept for backward-compat."""
    return planck_temp(band, rad)


def planck_deriv_T(band: int, T: float, p: float) -> float:
    """
    Partial derivative of p * L_band(T) with respect to T.
    Used in Newton-Raphson Jacobian for the Dozier method.

    d/dT [p * B(lambda, T)] = p * c1*c2 / (lambda^6 * T^2) *
                               exp(c2/(lambda*T)) / (exp(c2/(lambda*T))-1)^2

    Parameters
    ----------
    band : int
    T    : float [K]
    p    : float  fraction on fire

    Returns
    -------
    float
    """
    lam = LAMBDA[band]
    c1  = 2.0 * H_PLANCK * C_LIGHT**2
    c2  = H_PLANCK * C_LIGHT / K_BOLTZ
    x   = c2 / (lam * T)
    ex  = np.exp(x)
    return p * c1 * c2 / (lam**6 * T**2) * ex / (ex - 1.0)**2
