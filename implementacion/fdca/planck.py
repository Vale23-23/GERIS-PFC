"""
Planck / radiometry utilities for FDCA

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

def planck_temp_from_coeffs(rad, fk1, fk2, bc1, bc2):
    """
    Inversión oficial NOAA (ABI L1b PUG, sección 4.2.4): radiancia cruda
    del .nc -> temperatura de brillo [K].

    rad debe ser la radiancia tal cual viene en la variable 'Rad' del .nc,
    SIN ninguna conversión de unidades manual. fk1, fk2, bc1, bc2 son los
    coeficientes propios de ESA banda y ESE archivo (planck_fk1, planck_fk2,
    planck_bc1, planck_bc2), no constantes genéricas de constants.py.

    Fórmula:
        BT = (fk2 / ln(fk1/rad + 1) - bc1) / bc2

    Parameters
    ----------
    rad : array-like
        Radiancia cruda [mW m-2 sr-1 (cm-1)-1]
    fk1, fk2, bc1, bc2 : float
        Coeficientes de calibración leídos del .nc (o del *_planck.json)

    Returns
    -------
    ndarray [K]
    """
    rad = np.asarray(rad, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        BT = (fk2 / np.log1p(fk1 / np.where(rad > 0, rad, np.nan)) - bc1) / bc2
    return BT

def planck_rad_from_coeffs( T, fk1_target, fk2_target, bc1_target, bc2_target):
    """
    Inversión de planck_temp_from_coeffs: temperatura de brillo -> radiancia
    cruda, en la unidad nativa del INSTRUMENTO/BANDA cuyos coeficientes se
    pasan (mW m-2 sr-1 (cm-1)-1), no en la unidad genérica de planck_rad().

    Se usa para responder: "si el canal `band_target` hubiera visto un
    cuerpo negro a temperatura T, ¿qué radiancia cruda habría medido?"
    usando los coeficientes reales de ESE canal (fk1_target, fk2_target,
    bc1_target, bc2_target), no los de la banda de origen de T.

    Fórmula (despejada de planck_temp_from_coeffs):
        L = fk1 / (exp(fk2 / (BT*bc2 + bc1)) - 1)

    Parameters
    ----------
    
    T : array-like [K]
        Temperatura de brillo de entrada (puede ser de OTRA banda, ej. BT14).
    fk1_target, fk2_target, bc1_target, bc2_target : float
        Coeficientes Planck de la banda EN LA QUE querés expresar la
        radiancia resultante (ej. coeffs7 si querés "BT14 visto en rad7").

    Returns
    -------
    ndarray
        Radiancia en la unidad nativa del archivo .nc de esa banda
        [mW m-2 sr-1 (cm-1)-1].
    """
    T = np.asarray(T, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        exponent = fk2_target / (T * bc2_target + bc1_target)
        L = fk1_target / (np.expm1(exponent))
    return L

def planck_deriv_T_from_coeffs(T, p: float, fk1, fk2, bc1, bc2):
    """
    Derivada de p * L(T) respecto de T, usando la MISMA parametrización
    de planck_rad_from_coeffs (coeficientes reales del instrumento), no la
    fórmula de Planck genérica de planck_deriv_T.

    Es el Jacobiano que necesita el Newton-Raphson de dozier.py para
    trabajar en la unidad nativa del sensor (mW m-2 sr-1 (cm-1)-1) en vez
    de la unidad genérica W m-2 sr-1 m-1 que usaba planck_deriv_T.

    Derivación (a partir de L(T) = fk1 / (exp(x) - 1), con
    x = fk2 / (T*bc2 + bc1)):

        dx/dT   = -fk2*bc2 / (T*bc2 + bc1)^2
        dL/dT   = -fk1 * exp(x) * dx/dT / (exp(x) - 1)^2
                =  fk1 * fk2 * bc2 * exp(x) / [(T*bc2+bc1)^2 * (exp(x)-1)^2]

    Parameters
    ----------
    T : float [K]
    p : float  fracción de fuego en el píxel
    fk1, fk2, bc1, bc2 : float  coeficientes Planck de la banda en la que
        se está evaluando (misma banda que se usó para L(T) en el resto
        del sistema de ecuaciones de Dozier).

    Returns
    -------
    float
    """
    denom = T * bc2 + bc1
    x = fk2 / denom
    with np.errstate(over="ignore"):
        ex = np.exp(x)
    dLdT = fk1 * fk2 * bc2 * ex / (denom**2 * (ex - 1.0)**2)
    return p * dLdT