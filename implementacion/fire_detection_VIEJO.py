"""
fire_detection.py — Algoritmo de detección de fuego NOAA/GOES (Fase 1).

Implementa las secciones 3.4.2.2 – 3.4.2.13 del algoritmo NOAA para
detección sub-píxel de fuego con datos ABI.  Sigue fielmente el
pseudocódigo (pseudocodigo_parte1_ver2.pseudo) y el patrón del resto
del pipeline (downloader.py, static_data_downloader.py).

Todas las temperaturas están en Kelvin; todos los ángulos en grados.

Entradas (arrays 2D [L, W] en float32 salvo indicación contraria):
    bTemp7, rad7      — Temperatura de brillo / radianza banda 7  (3.9 µm)
    bTemp13, rad13    — Idem banda 13 (10.3 µm)
    bTemp14, rad14    — Idem banda 14 (11.2 µm)
    bTemp15, rad15    — Idem banda 15 (12.3 µm)  [opcional, puede ser None]
    refl2             — Reflectancia banda 2 (0.62 µm)           [opcional]
    latitudes, longitudes
    ang_cenital_sol   — Ángulo cenital solar (deg)
    ang_resp_sol      — Ángulo de resplandor solar (deg)
    ang_cenital_local — Ángulo cenital local (deg)
    azimut_relativo   — (deg, reservado para uso futuro)
    angulo_relativo   — (deg, reservado para uso futuro)
    data_quality      — Mapa de calidad de datos del ABI
    tpw               — Agua precipitable total (mm), modelo GFS/NCEP
    emisividad7       — Emisividad superficie banda 7
    emisividad14      — Emisividad superficie banda 14
    mascara_fuego_anterior — Detección de fuego del paso anterior
    cobertura_terrestre    — Clases GlobCover (int16)
    mascara_tierra_mar     — 1=tierra, 0=mar (int8)
    mascara_desierto       — 1=desierto (no usada en Uruguay)
    LUT_TPW                — np.ndarray shape (6, 35+): ver cabecera del pseudo
    FPT                    — Temperatura del plano focal (float, escalar)

Salidas (dict):
    "mascara_fuego"   — Códigos de píxel (uint8/int16)
    "fail_char"       — Flags de fallo por píxel (int8)
    "fire_size"       — Tamaño estimado del fuego sub-píxel (float32)
    "fire_temperature"— Temperatura estimada del fuego (float32)
    "pixel_area"      — Área del píxel (float32, a completar en Fase 2)
    "frp"             — Fire Radiative Power (float32, a completar en Fase 2)
    "candidatos"      — Lista de (i, j) de píxeles candidatos a fuego

Códigos de mascara_fuego (todos los presentes en el pseudo):
    100  — Sin fuego / pixel válido
    40   — Píxel de espacio (fuera del disco terrestre)
    50   — Ángulo cenital solar > 80° (limbo)
    60   — Ángulo cenital solar < 10° ó ángulo resplandor < 10°
    120  — Missing value en bTemp7
    121  — Missing value en bTemp14
    123  — Saturación bTemp7
    124  — Saturación bTemp14
    125  — Radianza negativa
    126  — bTemp7 < 200 K (inutilizable)
    127  — bTemp14 < 200 K (inutilizable)
    170  — Ventana de fondo no convergió (< 20 % de píxeles válidos)
    180  — Radianzas/temperaturas corregidas negativas (TPW o emisividad)
    200  — Nube: bTemp14 < 270 K
    201  — Píxel demasiado frío (abs(T7-T14) < 2 y alguno ≤ 273 K)
    205  — Nube: bTemp7 - bTemp14 < -4
    215  — Nube: albedo > 0.38 (día)
    220  — Nube: bTemp15 ≤ 265 K
    225  — Nube: bTemp14 - bTemp15 < -4
    230  — Nube: bTemp14 - bTemp15 > 60 (posible inverso)
    240  — Borde de nube nocturno (along-scan, descartado)
    245  — Borde de nube diurno (along-scan, descartado)
    FUEGO — ver flags de fail_char (positivos)

fail_char (flags acumulados, -1 = no evaluado):
    -1   — No evaluado
     7   — Píxel saturado (bTemp7 ≥ 411.76 K ó bTemp14 ≥ 339.9 K)
     1   — Fallo test T7-T14 < umbral_7b_14b (§3.4.2.7)
     2   — Fallo test T7-Tbkg < umbral_7b (§3.4.2.7)
     3   — Post-corrección: T14_corr < 285 ó T7_corr < 285+offset (§3.4.2.9)
     4   — Post-corrección: T14_corr - Tbkg_corr < 0.25 (§3.4.2.9)
     5   — Post-corrección: T7_corr - Tbkg_corr < 2 (§3.4.2.9)
    10   — Post-corrección: albedo > 0.15 ó nube con T7_corr-Tbkg_corr > 10 (§3.4.2.9)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Constantes globales
# ══════════════════════════════════════════════════════════════════════════════

BR_TEMP_MISS_VAL: float = -999.0   # Valor centinela de temperatura de brillo faltante
SAT_TEMP: float          = 335.0   # Temperatura de saturación del sensor (K)
SAT_BUFF: float          = 5.0     # Buffer de saturación (K)
FPT_THRESHOLD: float     = 90.0    # Umbral de temperatura del plano focal

# Constantes de Planck para cada banda ABI usada
# Basadas en los coeficientes espectrales del ABI (GOES-R PUG L1b Rev 2)
# c1 = 2*h*c^2  [mW / (m^2·sr·cm^-4)]
# c2 = h*c/k    [K·cm]
_ABI_BAND_PARAMS: dict[int, dict[str, float]] = {
    7:  {"fk1": 202526.8, "fk2": 3698.03, "bc1": 0.4759,  "bc2": 0.99810},
    13: {"fk1":  671922.5, "fk2": 1285.62, "bc1": 0.3764,  "bc2": 0.99891},
    14: {"fk1":  620818.5, "fk2": 1216.99, "bc1": 0.3572,  "bc2": 0.99912},
    15: {"fk1":  520364.0, "fk2": 1107.43, "bc1": 0.3011,  "bc2": 0.99933},
}


# ══════════════════════════════════════════════════════════════════════════════
# Funciones auxiliares de radiometría (Planck)
# ══════════════════════════════════════════════════════════════════════════════

def _rad_from_temp(band: int, temp: np.ndarray | float) -> np.ndarray | float:
    """
    Temperatura de brillo → radianza equivalente para *band*.
    Usa la función de Planck inversa con los coeficientes del ABI.
    """
    p   = _ABI_BAND_PARAMS[band]
    t   = (temp - p["bc1"]) / p["bc2"]
    # Evitar división por cero / overflow en exp
    t   = np.where(t > 0, t, np.nan) if isinstance(t, np.ndarray) else max(t, 1e-6)
    rad = p["fk1"] / (np.exp(p["fk2"] / t) - 1.0)
    return rad


def _temp_from_rad(band: int, rad: np.ndarray | float) -> np.ndarray | float:
    """
    Radianza → temperatura de brillo equivalente para *band*.
    """
    p    = _ABI_BAND_PARAMS[band]
    safe = np.where(rad > 0, rad, np.nan) if isinstance(rad, np.ndarray) else max(rad, 1e-12)
    t    = p["fk2"] / np.log(p["fk1"] / safe + 1.0)
    return p["bc1"] + p["bc2"] * t


def temp_to_rad(src_band: int, dst_band: int, temp: np.ndarray | float) -> np.ndarray | float:
    """
    Dada una temperatura de brillo en src_band, calcula la radianza
    equivalente en dst_band mediante la función de Planck.

    El flujo es:  T_brillo(src) → L(src) → T_física → L(dst).
    En la práctica, para los tests diferenciales del algoritmo NOAA,
    src_band y dst_band suelen diferir (ej. convertir T14 a radianza de B7).
    """
    # Paso 1: T_brillo(src) → radianza(src)
    rad_src = _rad_from_temp(src_band, temp)
    if src_band == dst_band:
        return rad_src
    # Paso 2: radianza(src) → T_física → radianza(dst)
    #   Usamos la temperatura de brillo de src como aproximación de T_física
    return _rad_from_temp(dst_band, temp)


def rad_to_temp(src_band: int, dst_band: int, rad: np.ndarray | float) -> np.ndarray | float:
    """
    Dada una radianza en src_band, devuelve la temperatura de brillo
    equivalente en dst_band.
    """
    temp_src = _temp_from_rad(src_band, rad)
    if src_band == dst_band:
        return temp_src
    return _temp_from_rad(dst_band, _rad_from_temp(src_band, temp_src))


# ══════════════════════════════════════════════════════════════════════════════
# LUT TPW — acceso
# ══════════════════════════════════════════════════════════════════════════════

def _lut_lookup(lut: np.ndarray, line: int, column: int) -> float:
    """
    Acceso a la LUT de TPW.

    Estructura de la LUT (ver cabecera del pseudocódigo):
        Filas  (0-5): TPW bin, offset radianza, trans_7, trans_14, ext_7, ext_14
        Columnas    : 7 bins de ángulo cenital × 5 bins de TPW = 35 columnas
                      (índice base 1 en el pseudo → base 0 aquí)
    """
    return float(lut[line, column - 1])   # el pseudo usa índices base-1


def _lut_indices(tpw: float, ang_cenital_local_deg: float) -> tuple[int, int]:
    """
    Devuelve (bin_tpw, indice_col) para la LUT, con clamp a rango válido.

    bin_tpw ∈ [1, 5]  — cada 10 mm de TPW
    bin_ang  ∈ [1, 7]  — cada 10° de ángulo cenital local
    indice_col = 7*(bin_tpw - 1) + bin_ang
    """
    bin_tpw = int(round(tpw / 10.0))
    bin_tpw = max(1, min(5, bin_tpw))

    bin_ang = int(round(ang_cenital_local_deg / 10.0))
    bin_ang = max(1, min(7, bin_ang))

    indice_col = 7 * (bin_tpw - 1) + bin_ang
    return bin_tpw, indice_col


# ══════════════════════════════════════════════════════════════════════════════
# Función principal
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1(
    # ── Bandas ABI ──────────────────────────────────────────────────────────
    bTemp7:  np.ndarray,
    rad7:    np.ndarray,
    bTemp13: np.ndarray,
    rad13:   np.ndarray,
    bTemp14: np.ndarray,
    rad14:   np.ndarray,
    bTemp15: Optional[np.ndarray],
    rad15:   Optional[np.ndarray],
    refl2:   Optional[np.ndarray],
    # ── Geometría y auxiliares dinámicos ────────────────────────────────────
    latitudes:           np.ndarray,
    longitudes:          np.ndarray,
    ang_cenital_sol:     np.ndarray,
    ang_resp_sol:        np.ndarray,
    ang_cenital_local:   np.ndarray,
    azimut_relativo:     np.ndarray,   # reservado
    angulo_relativo:     np.ndarray,   # reservado
    data_quality:        np.ndarray,
    tpw:                 np.ndarray,
    emisividad7:         np.ndarray,
    emisividad14:        np.ndarray,
    mascara_fuego_anterior: np.ndarray,
    # ── Datos estáticos ─────────────────────────────────────────────────────
    cobertura_terrestre: np.ndarray,
    mascara_tierra_mar:  np.ndarray,
    mascara_desierto:    np.ndarray,
    LUT_TPW:             np.ndarray,
    # ── Escalar ─────────────────────────────────────────────────────────────
    FPT: float,
) -> dict:
    """
    Ejecuta la Fase 1 del algoritmo de detección de fuego.

    Retorna un diccionario con las salidas descritas en el módulo docstring.
    """
    L, W = bTemp7.shape

    # ── Helpers locales ───────────────────────────────────────────────────────

    def es_dia(i: int, j: int) -> bool:
        """§3.4.2.2 — Píxel de día: 0 ≤ ang_cenital_sol ≤ 85°"""
        return 0.0 <= ang_cenital_sol[i, j] <= 85.0

    def es_espacio(i: int, j: int) -> bool:
        """
        Detecta si el píxel cae fuera del disco terrestre.
        TODO: el pseudo deja esto como "¿Píxel de espacio??" sin definición
        explícita. La forma estándar es verificar si bTemp7 y bTemp14 son
        ambos NaN o tienen el missing value específico del sensor.
        """
        return (
            bTemp7[i, j] == BR_TEMP_MISS_VAL
            and bTemp14[i, j] == BR_TEMP_MISS_VAL
        )

    # ── §3.4.2.2: Preprocesamiento — diferencias de radianza ─────────────────

    logger.info("§3.4.2.2 — Preprocesamiento: diferencias de radianza")

    rad14_a_7 = temp_to_rad(14, 7, bTemp14)   # radianzas de B14 pasadas a B7
    rad13_a_7 = temp_to_rad(13, 7, bTemp13)   # radianzas de B13 pasadas a B7

    refl_7_14 = rad7 - rad14_a_7              # diferencia B7 - B14 (en escala B7)
    refl_7_13 = rad7 - rad13_a_7              # diferencia B7 - B13

    # Si la temperatura del plano focal es demasiado alta, se crea una banda
    # híbrida que toma el mínimo de los dos deltas para mitigar la distorsión.
    if FPT > FPT_THRESHOLD:
        logger.warning("FPT=%.1f > %.1f — usando banda híbrida 7/13", FPT, FPT_THRESHOLD)
        refl = np.minimum(np.abs(refl_7_14), np.abs(refl_7_13))
    else:
        refl = refl_7_14

    # ── Inicialización de máscaras de salida ──────────────────────────────────

    mascara_fuego   = np.full((L, W), 100, dtype=np.int16)
    fail_char       = np.full((L, W), -1,  dtype=np.int8)
    mascara_nubes   = np.zeros((L, W), dtype=bool)
    mascara_saturados = np.zeros((L, W), dtype=bool)
    fire_size       = np.zeros((L, W), dtype=np.float32)
    pixel_area      = np.zeros((L, W), dtype=np.float32)
    fire_temperature = np.full((L, W), np.nan, dtype=np.float32)
    frp             = np.full((L, W), np.nan, dtype=np.float32)

    for i in range(L):
        for j in range(W):
            if es_espacio(i, j):
                mascara_fuego[i, j] = 40
            # (los demás arrancan en 100 por el full)

    # ── Albedo e intensidad visible ───────────────────────────────────────────

    albedo       = np.full((L, W), np.nan, dtype=np.float32)
    vis_brightness = np.full((L, W), np.nan, dtype=np.float32)

    if refl2 is not None:
        logger.info("Calculando albedo y vis_brightness (banda 2 disponible)")
        cos_sza = np.cos(np.radians(ang_cenital_sol))
        # Evitar división por cero en ángulos muy altos
        cos_sza_safe = np.where(np.abs(cos_sza) > 1e-6, cos_sza, np.nan)
        albedo        = (refl2 / cos_sza_safe).astype(np.float32)
        vis_brightness = (255 * np.sqrt(np.maximum(refl2, 0))).astype(np.float32)

    # ── §3.4.2.3: Loop principal sobre todos los píxeles ─────────────────────

    logger.info("§3.4.2.3 — Loop principal (%d × %d píxeles)", L, W)

    cantidad_candidatos = 0
    candidatos: list[tuple[int, int]] = []

    for i in range(L):
        for j in range(W):

            # ── Tests iniciales (descarte rápido) ─────────────────────────────

            if ang_cenital_sol[i, j] > 80:
                mascara_fuego[i, j] = 50
                continue

            if ang_cenital_sol[i, j] < 10 or ang_resp_sol[i, j] < 10:
                mascara_fuego[i, j] = 60
                continue

            if bTemp7[i, j] == BR_TEMP_MISS_VAL:
                mascara_fuego[i, j] = 120
                continue

            if bTemp14[i, j] == BR_TEMP_MISS_VAL:
                mascara_fuego[i, j] = 121
                continue

            if bTemp7[i, j] > SAT_TEMP + SAT_BUFF:
                mascara_fuego[i, j] = 123
                continue

            if bTemp14[i, j] > SAT_TEMP + SAT_BUFF:
                mascara_fuego[i, j] = 124
                continue

            if bTemp7[i, j] < 200:
                mascara_fuego[i, j] = 126
                continue

            if bTemp14[i, j] < 200:     # pseudo dice "bTemp14 z 200" (typo → <)
                mascara_fuego[i, j] = 127
                continue

            # TODO §3.4.2.3 — Test de ecosistemas (no definido en el pseudo)

            # ── Mínimo umbral para actividad de fuego ─────────────────────────

            if rad7[i, j] < 0 or rad14[i, j] < 0:
                mascara_fuego[i, j] = 125
                continue

            # Píxel demasiado tibio / sin diferencia apreciable
            if (bTemp7[i, j] > 273 or bTemp14[i, j] > 273) and (
                abs(bTemp7[i, j] - bTemp14[i, j]) <= 2
            ):
                mascara_fuego[i, j] = 100   # descartado como fuego, pero válido
                continue

            if abs(bTemp7[i, j] - bTemp14[i, j]) < 2 and (
                bTemp7[i, j] <= 273 or bTemp14[i, j] <= 273
            ):
                mascara_fuego[i, j] = 201   # píxel demasiado frío
                # No hay continue aquí en el pseudo; el píxel sigue evaluándose

            # ── Umbrales dependientes del ciclo día/noche ─────────────────────

            offset_7      = 0.0
            offset_7_refl = 0.0

            if es_dia(i, j):
                cos_sza_ij     = np.cos(np.radians(ang_cenital_sol[i, j]))
                offset_7       = 15.0 * cos_sza_ij
                offset_7_refl  = 5.0  * cos_sza_ij

            bTemp7_min             = 285.0 + offset_7
            bTemp7_refl_threshold  = 315.0 + offset_7_refl

            # ── §3.4.2.3: Test de nubes ───────────────────────────────────────

            if mascara_fuego[i, j] == 100:

                if bTemp14[i, j] < 270:
                    mascara_fuego[i, j] = 200
                    mascara_nubes[i, j] = True
                    continue

                if bTemp7[i, j] - bTemp14[i, j] < -4:
                    mascara_fuego[i, j] = 205
                    mascara_nubes[i, j] = True
                    continue

                if es_dia(i, j):
                    sza = ang_cenital_sol[i, j]
                    lza = ang_cenital_local[i, j]
                    # Condición de umbral angular para test de albedo
                    if sza <= 70 or (sza <= 60 and lza <= 60):
                        if not np.isnan(albedo[i, j]) and albedo[i, j] > 0.38:
                            mascara_fuego[i, j] = 215
                            mascara_nubes[i, j] = True
                            continue

                if bTemp15 is not None:
                    if bTemp15[i, j] <= 265:
                        mascara_fuego[i, j] = 220
                        mascara_nubes[i, j] = True
                        continue

                    if bTemp14[i, j] < 270:
                        diff_14_15 = bTemp14[i, j] - bTemp15[i, j]
                        if diff_14_15 < -4:
                            mascara_fuego[i, j] = 225
                            mascara_nubes[i, j] = True
                            continue
                        if diff_14_15 > 60:
                            mascara_fuego[i, j] = 230
                            mascara_nubes[i, j] = True
                            continue

            # ── §3.4.2.4: Along-scan reflectivity test (bordes de nubes) ──────
            # NOTA del pseudo: "hay que cuidar el loop en los bordes de la imagen"

            if es_dia(i, j):
                # Verificamos que tengamos columnas j-3 y j+3 accesibles
                j_left  = j - 3
                j_right = j + 3

                if j_left >= 0 and j_right < W:
                    if albedo[i, j] >= 0.38 and bTemp7[i, j] < 230:
                        if refl[i, j_left] < 0.2 or refl[i, j_right] < 0.2:
                            mascara_fuego[i, j] = 245
                            continue
                # En los bordes (j < 3 ó j >= W-3) se omite el test
                # para no salir de la imagen (ver nota del pseudo).
            else:
                j_left  = j - 3
                j_right = j + 3

                if j_left >= 0 and j_right < W:
                    if (
                        bTemp7[i, j] < bTemp7_min
                        and bTemp7[i, j] > 150
                        and (refl[i, j_left] < 0.2 or refl[i, j_right] < 0.2)
                    ):
                        mascara_fuego[i, j] = 240
                        continue

            # ── Test de saturación ────────────────────────────────────────────

            if bTemp7[i, j] >= 411.76 or bTemp14[i, j] >= 339.9:
                mascara_saturados[i, j] = True
                fail_char[i, j] = 7

            # ── §3.4.2.5: Estadísticas de fondo (ventana expansiva) ───────────

            tam_ventana = 5     # semiancho inicial → ventana 11×11
            fin         = False
            iteraciones = 0

            while not fin and iteraciones < 10:
                pixeles_validos = 0
                total_ventana = (2 * tam_ventana + 1) ** 2

                i_min = max(0,   i - tam_ventana)
                i_max = min(L-1, i + tam_ventana)
                j_min = max(0,   j - tam_ventana)
                j_max = min(W-1, j + tam_ventana)

                for x in range(i_min, i_max + 1):
                    for y in range(j_min, j_max + 1):
                        if x == i and y == j:
                            continue   # excluir el píxel central

                        sza_xy = ang_cenital_sol[x, y]
                        cos_sza_xy = np.cos(np.radians(sza_xy))
                        if es_dia(x, y):
                            muy_calido = bTemp7[x, y] > 310 + 25 * cos_sza_xy
                        else:
                            muy_calido = bTemp7[x, y] > 310

                        muy_frio = bTemp7[x, y] < 270 and bTemp14[x, y] < 270

                        vb_xy  = vis_brightness[x, y]
                        alb_xy = albedo[x, y]

                        if (
                            mascara_tierra_mar[x, y] == 1
                            and not muy_calido
                            and not muy_frio
                            and (not np.isnan(vb_xy) and vb_xy >= 1)
                            and (not np.isnan(alb_xy) and alb_xy < 0.38)
                        ):
                            pixeles_validos += 1

                # El pseudo cuenta el total con los píxeles que se saldrían de
                # la imagen (total_ventana), no solo los que están dentro.
                if pixeles_validos >= 0.2 * total_ventana:
                    fin = True
                else:
                    iteraciones += 1
                    if not fin:
                        tam_ventana += 5

            if not fin:
                mascara_fuego[i, j] = 170
                continue

            # ── Extraer ventana de fondo ──────────────────────────────────────

            i_min = max(0,   i - tam_ventana)
            i_max = min(L-1, i + tam_ventana) + 1
            j_min = max(0,   j - tam_ventana)
            j_max = min(W-1, j + tam_ventana) + 1

            win_b7   = bTemp7[i_min:i_max, j_min:j_max]
            win_b14  = bTemp14[i_min:i_max, j_min:j_max]
            win_refl = refl[i_min:i_max, j_min:j_max]
            win_vis  = vis_brightness[i_min:i_max, j_min:j_max]

            temp7_bkg_mean  = float(np.nanmean(win_b7))
            temp14_bkg_mean = float(np.nanmean(win_b14))
            vis_mean_bkg    = float(np.nanmean(win_vis))

            temp7_bkg_std_dev  = float(np.nanstd(win_b7))
            temp14_bkg_std_dev = float(np.nanstd(win_b14))

            std_dev_7_14_diff = float(np.nanstd(win_b7 - win_b14))

            # reflb — reflectancia media de fondo en escala de rad7
            # TODO el pseudo dice "calcularlo fuera del loop principal"; aquí se
            # calcula en el loop porque depende de la ventana de cada píxel.
            reflb = float(np.nanmean(win_refl))

            # TODO: las siguientes estadísticas están marcadas como "..." en el
            # pseudo y requieren especificación adicional (histogramas de fondo):
            #   idx_cld_bkg, bkg_count_idx,
            #   temp4_bkg_histogram, temp11_bkg_histogram,
            #   temp4_bkg_histogram_stddev, temp11_bkg_histogram_stddev,
            #   vis_diff_histogram, vis_histogram_variance, vis_histogram_stddev,
            #   temp4_bkg_avg, temp4_stddev, temp11_bkg_avg, temp11_stddeve,
            #   rad_4mu_11mu_avg_diff, rad_diff_sigma
            # Se dejan como NaN hasta que se defina la especificación completa.
            number_passes_in_bkg_statistics = iteraciones

            # ── §3.4.2.6: Umbrales contextuales ──────────────────────────────

            offset_ventana = min(5.0, number_passes_in_bkg_statistics / 3.0)

            std_dev_7b_14b_test_threshold = min(
                4.0, 3.0 * std_dev_7_14_diff
            )
            std_dev_7b_test_threshold = min(
                10.0, max(4.0, 3.75 * temp7_bkg_std_dev + offset_ventana)
            )
            std_dev_reflb_test_threshold = min(
                1.0, min(0.25, 3.0 * float(np.nanstd(win_refl)))
            )

            # TODO: std_dev_refelb_max_test_threshold — el pseudo lo menciona
            # pero no da su fórmula explícita.  Se usa el mismo valor que
            # std_dev_reflb_test_threshold como aproximación conservadora.
            std_dev_refelb_max_test_threshold = std_dev_reflb_test_threshold

            # TODO: along_scan_line_test — no especificado en la Fase 1.
            # Se marca como False hasta que se implemente.
            along_scan_line_test = False

            # Albedo de fondo (solo día)
            albedo_bkg = np.nan
            if es_dia(i, j) and not np.isnan(vis_brightness[i, j]):
                vis_ij = vis_brightness[i, j]
                cos_sza_ij = np.cos(np.radians(ang_cenital_sol[i, j]))
                if cos_sza_ij > 1e-6:
                    albedo_bkg = ((vis_ij / 25.5) ** 2) / (100.0 * cos_sza_ij)

            # ── §3.4.2.7: Aplicación de umbrales para identificar fuego ───────

            # Descartar como fuego si no supera ninguno de los dos umbrales
            if (
                bTemp7[i, j] - bTemp14[i, j] < std_dev_7b_14b_test_threshold
                and
                bTemp7[i, j] - temp7_bkg_mean < std_dev_7b_test_threshold
            ):
                continue   # No es candidato a fuego

            fire_size[i, j]     = 0.0
            pixel_area[i, j]    = 0.0

            if mascara_saturados[i, j]:
                fire_temperature[i, j] = 0.0
            else:
                fire_temperature[i, j] = -9.05

            # Tests adicionales de descarte
            if (
                (refl[i, j] < std_dev_reflb_test_threshold and bTemp7[i, j] < 320)
                and
                (
                    bTemp7[i, j] - bTemp14[i, j] < 0
                    or bTemp7[i, j] - temp7_bkg_mean < 0
                )
            ):
                continue   # Descartado como fuego sin flag

            if (
                bTemp7[i, j] - bTemp14[i, j] < std_dev_7b_14b_test_threshold
            ) and (
                refl[i, j] < std_dev_refelb_max_test_threshold
                or along_scan_line_test
            ):
                fail_char[i, j] = 1   # flag = 1

            if (
                bTemp7[i, j] - temp7_bkg_mean < std_dev_7b_test_threshold
            ) and (
                refl[i, j] < std_dev_refelb_max_test_threshold
                or along_scan_line_test
            ):
                fail_char[i, j] = 2   # flag = 2

            # ── §3.4.2.8: Correcciones y ajustes ─────────────────────────────

            # Corrección de transmitancia atmosférica (LUT TPW)
            tpw_ij = float(tpw[i, j])
            acl_ij = float(ang_cenital_local[i, j])
            _, indice_col = _lut_indices(tpw_ij, acl_ij)

            ext7   = _lut_lookup(LUT_TPW, 4, indice_col)
            ext14  = _lut_lookup(LUT_TPW, 5, indice_col)
            trans7 = _lut_lookup(LUT_TPW, 2, indice_col)
            trans14 = _lut_lookup(LUT_TPW, 3, indice_col)

            rad7_ij  = float(temp_to_rad(7,  7,  bTemp7[i, j]))
            rad14_ij = float(temp_to_rad(14, 14, bTemp14[i, j]))

            # rad_corr = (rad - rad*ext) / trans
            if abs(trans7) < 1e-9 or abs(trans14) < 1e-9:
                mascara_fuego[i, j] = 180
                continue

            rad7_corr  = (rad7_ij  - rad7_ij  * ext7)  / trans7
            rad14_corr = (rad14_ij - rad14_ij * ext14) / trans14

            bTemp7_corr  = float(rad_to_temp(7,  7,  rad7_corr))
            bTemp14_corr = float(rad_to_temp(14, 14, rad14_corr))

            if rad7_corr < 0 or rad14_corr < 0 or bTemp7_corr < 0 or bTemp14_corr < 0:
                mascara_fuego[i, j] = 180
                continue

            # Corrección por nubes o humo semi-transparente (solo día)
            if es_dia(i, j) and not np.isnan(albedo_bkg):
                albedo_diff = albedo[i, j] - albedo_bkg
                if 0.025 < albedo_diff < 0.07:
                    bTemp7_corr  += 10.0 * albedo_diff
                    bTemp14_corr += 30.0 * albedo_diff
                    if albedo[i, j] > 0.38 or albedo_diff > 0.38:
                        bTemp7_corr  += 0.7
                        bTemp14_corr += 2.1
                        # TODO: setear flag indicando esta condición (§3.4.2.8)

            # Corrección por emisividad de la superficie
            emis7_ij  = float(emisividad7[i, j])
            emis14_ij = float(emisividad14[i, j])

            if emis7_ij < 1e-6 or emis14_ij < 1e-6:
                mascara_fuego[i, j] = 180
                continue

            rad7_corr_emiss  = rad7_corr  / emis7_ij
            rad14_corr_emiss = rad14_corr / emis14_ij

            # Corrección de reflectividad solar
            rad7_bkg_corr  = (temp7_bkg_mean  - temp7_bkg_mean  * ext7)  / trans7
            rad14_bkg_corr = (temp14_bkg_mean - temp14_bkg_mean * ext14) / trans14

            # rad7from14: radianza de B7 a partir de la temperatura de B14 corregida
            temp14_corr_emiss = float(rad_to_temp(14, 14, rad14_corr_emiss))
            rad7from14        = float(temp_to_rad(14, 7, temp14_corr_emiss))
            rad7from14_bkg_corr = (rad7from14 - rad7from14 * ext7) / trans7

            rad_solar      = rad7_bkg_corr - emis7_ij * rad7from14_bkg_corr
            rad7_corr_solar = (rad7_corr_emiss - rad_solar) / emis7_ij

            # TODO §3.4.2.8 — verificar si alguna temp corregida < 0
            # (el pseudo dice "si alguna de las temps corregidas < 0")
            if rad7_corr_solar < 0:
                mascara_fuego[i, j] = 180
                continue

            # Corrección por difracción (scattering)
            rad7_diff  = (rad7_corr_solar     - 0.15 * rad7from14_bkg_corr)  / 0.85
            rad14_diff = (rad14_corr          - 0.30 * rad14_bkg_corr)       / 0.70

            # Convertir de vuelta a temperatura
            bTemp7_corr  = float(rad_to_temp(7,  7,  rad7_diff))
            bTemp14_corr = float(rad_to_temp(14, 14, rad14_diff))

            # TODO §3.4.2.8 — flag = 8 (no especificado en el pseudo)
            # bTemp_bkg_corr: el pseudo sugiere rad_to_temp(7, 7, rad7_bkg_corr)
            bTemp_bkg_corr = float(rad_to_temp(7, 7, rad7_bkg_corr))

            # ── §3.4.2.9: Tests post-corrección ───────────────────────────────

            offset_dia = 0.0
            if es_dia(i, j):
                offset_dia = 15.0 * np.cos(np.radians(ang_cenital_sol[i, j]))

            if bTemp14_corr < 285 or bTemp7_corr < 285 + offset_dia:
                fail_char[i, j] = 3

            if bTemp14_corr - bTemp_bkg_corr < 0.25:
                alb_ij = albedo[i, j]
                if (
                    (not np.isnan(alb_ij) and alb_ij > 0.15)
                    or mascara_nubes[i, j]
                ) and bTemp7_corr - bTemp_bkg_corr > 10:
                    fail_char[i, j] = 10
                else:
                    fail_char[i, j] = 4

            if bTemp7_corr - bTemp_bkg_corr < 2:
                fail_char[i, j] = 5

            # ── §3.4.2.10: Caracterización sub-píxel (Dozier) ─────────────────
            # TODO: implementar el método de Dozier para estimar temperatura y
            # área del fuego.  Requiere las dos ecuaciones de Planck simultáneas
            # (B7 y B14) y la FRP.  Se deja como placeholder.

            # ── §3.4.2.11: Last-chance fire tests ─────────────────────────────

            refl_excess = refl[i, j] - reflb

            if (
                (bTemp7[i, j] - temp7_bkg_mean >= std_dev_7b_test_threshold
                 and bTemp14[i, j] - temp14_bkg_mean >= -20)
                or
                (refl_excess >= std_dev_refelb_max_test_threshold
                 and along_scan_line_test)
            ):
                fire_size[i, j] = 0.0

                # Temperatura del fuego: si está en rango marginal, negativa
                if 350.0 < fire_temperature[i, j] <= 400.0:
                    fire_temperature[i, j] *= -1.0
                else:
                    fire_temperature[i, j] = -999.0

            # TODO §3.4.2.12: FRP (Fire Radiative Power) — requires Dozier output

            # ── §3.4.2.13: Fin Fase 1 — acumular candidatos ───────────────────

            # Criterio de candidato: el píxel llegó hasta aquí sin continue Y
            # al menos uno de los umbrales principales fue superado.
            # TODO: el pseudo dice "if(#Potencial pixel de fuego)" sin definición
            # exacta del predicado.  Se usa fail_char != -1 como proxy.
            es_candidato = fail_char[i, j] != -1

            if es_candidato:
                mascara_fuego[i, j] = 10   # código provisional de "fuego detectado"
                cantidad_candidatos += 1
                candidatos.append((i, j))

    logger.info("Fase 1 completa — candidatos a fuego: %d", cantidad_candidatos)

    return {
        "mascara_fuego":    mascara_fuego,
        "fail_char":        fail_char,
        "fire_size":        fire_size,
        "fire_temperature": fire_temperature,
        "pixel_area":       pixel_area,
        "frp":              frp,
        "candidatos":       candidatos,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de I/O — integración con el pipeline de downloader.py
# ══════════════════════════════════════════════════════════════════════════════

def load_inputs(
    timestamp_str: str,
    product_dirs: dict[str, str],
    static_dir: str = "data/static",
    tpw_dir:    str = "data/tpw",
) -> dict:
    """
    Carga todos los arrays de entrada para un timestamp dado.

    Parameters
    ----------
    timestamp_str : str — formato "YYYYMMDD_HHMM" (igual que downloader.py)
    product_dirs  : dict que mapea product_id → carpeta donde están los .npy
                    Esperado al menos:
                        "ABI-L1b-Rad-B07", "ABI-L1b-Rad-B13",
                        "ABI-L1b-Rad-B14", "ABI-L1b-Rad-B15" (opcional),
                        "ABI-L1b-Ref-B02" (opcional),
                        "ABI-L2-angles"   (cenitales/azimutal),
                        "ABI-L2-FDCF"     (máscara fuego anterior)
    static_dir    : carpeta de datos estáticos (static_data_downloader.py)
    tpw_dir       : carpeta de datos TPW (tpw_downloader.py)

    Returns
    -------
    dict listo para desempaquetar en run_phase1(**inputs)
    """
    fname = f"{timestamp_str}.npy"

    def _load(product_id: str, required: bool = True) -> Optional[np.ndarray]:
        folder = product_dirs.get(product_id)
        if folder is None:
            if required:
                raise FileNotFoundError(f"product_dirs no contiene '{product_id}'")
            return None
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Archivo no encontrado: {path}")
            return None
        return np.load(path)

    import os

    # Bandas requeridas
    bTemp7  = _load("ABI-L1b-BT-B07")
    rad7    = _load("ABI-L1b-Rad-B07")
    bTemp13 = _load("ABI-L1b-BT-B13")
    rad13   = _load("ABI-L1b-Rad-B13")
    bTemp14 = _load("ABI-L1b-BT-B14")
    rad14   = _load("ABI-L1b-Rad-B14")

    # Bandas opcionales
    bTemp15 = _load("ABI-L1b-BT-B15", required=False)
    rad15   = _load("ABI-L1b-Rad-B15", required=False)
    refl2   = _load("ABI-L1b-Ref-B02", required=False)

    # Geometría y calidad
    ang_cenital_sol   = _load("ABI-L2-SZA")
    ang_resp_sol      = _load("ABI-L2-GlintAngle")
    ang_cenital_local = _load("ABI-L2-LZA")
    azimut_relativo   = _load("ABI-L2-AzimuthRelative")
    angulo_relativo   = _load("ABI-L2-AngleRelative")
    latitudes         = _load("ABI-L2-lat")
    longitudes        = _load("ABI-L2-lon")
    data_quality      = _load("ABI-L2-DQF")

    # Dinámicos
    tpw_path = os.path.join(tpw_dir, fname)
    tpw      = np.load(tpw_path) if os.path.exists(tpw_path) else np.zeros_like(bTemp7)

    emisividad7  = _load("emissivity_b07", required=False) or np.ones_like(bTemp7)
    emisividad14 = _load("emissivity_b14", required=False) or np.ones_like(bTemp7)

    mascara_fuego_anterior = _load("ABI-L2-FDCF", required=False) or np.zeros_like(bTemp7, dtype=np.int16)

    # Estáticos
    from static_data_downloader import load_static
    static = load_static(static_dir)

    return dict(
        bTemp7=bTemp7, rad7=rad7,
        bTemp13=bTemp13, rad13=rad13,
        bTemp14=bTemp14, rad14=rad14,
        bTemp15=bTemp15, rad15=rad15,
        refl2=refl2,
        latitudes=latitudes, longitudes=longitudes,
        ang_cenital_sol=ang_cenital_sol,
        ang_resp_sol=ang_resp_sol,
        ang_cenital_local=ang_cenital_local,
        azimut_relativo=azimut_relativo,
        angulo_relativo=angulo_relativo,
        data_quality=data_quality,
        tpw=tpw,
        emisividad7=emisividad7,
        emisividad14=emisividad14,
        mascara_fuego_anterior=mascara_fuego_anterior,
        cobertura_terrestre=static["land_cover"],
        mascara_tierra_mar=static["land_sea_mask"],
        mascara_desierto=np.zeros_like(static["land_sea_mask"]),  # no usada en Uruguay
        LUT_TPW=np.zeros((6, 35), dtype=np.float32),              # TODO: cargar LUT real
        FPT=0.0,   # TODO: leer del metadato del archivo ABI
    )