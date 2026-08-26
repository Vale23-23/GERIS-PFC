"""
Diagnóstico de falsos negativos LOW_PROB / TEMP_LOW por etapas.

Para cada escena:
  1. Identifica píxeles LOW_PROB y TEMP_LOW que no están en candidates.
  2. Calcula el estado (T7, T14, Tb7, Tb14) en cada etapa: crudo, TPW, emisividad, solar, difracción.
  3. Evalúa fail_char en cada etapa para encontrar exactamente dónde se pierde el píxel.
  4. Genera un reporte agregado del recall y el veredicto de run_part1.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

from fdca.constants import FireMask, FailChar
from fdca.constants import * 
from fdca.part1 import (
    run_part1, calculate_albedo,
    _tpw_lut_indices, _contextual_thresholds, _along_scan_reflectivity_test,
    _solar_correction, _background_albedo,
)
from fdca.planck import planck_rad_from_coeffs, planck_temp_from_coeffs
from fdca.background import compute_background
from fdca.fdca_adapter import load_fdca_input, build_tpw_lut


def _looks_like_repo_root(p: Path) -> bool:
    return (p / "implementacion").is_dir() and (p / "obtencion_imagenes").is_dir()

def _default_repo_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()
    for candidate in [start, *start.parents]:
        if _looks_like_repo_root(candidate):
            return candidate
    raise FileNotFoundError(
        f"No pude ubicar la raíz del repo GERIS-PFC.\n"
        'Definí GERIS_REPO_ROOT="/ruta/a/GERIS-PFC" y volvé a correr.'
    )

REPO_ROOT = Path(os.environ.get("GERIS_REPO_ROOT", _default_repo_root()))
IMPLEMENTATION_ROOT = REPO_ROOT / "implementacion"
DATASET_ROOT = REPO_ROOT / "obtencion_imagenes" / "dataset"
CONFIG_PATH = IMPLEMENTATION_ROOT / "fdca" / "config.yaml"
MASK_DIR = DATASET_ROOT / "uruguay" / "ABI-L2-FDCF-Mask"
eco_mask_path = IMPLEMENTATION_ROOT / "fdca" / "data" / "eco_mask.npy"
eco_mask = np.load(eco_mask_path).astype(np.uint8)

REGION = "uruguay"
DEBUG_TIMESTAMPS = ["20251207_0550"]
TARGET_CODES = {15: "LOW_PROB", 35: "TEMP_LOW"}
MAX_PIXELS_PER_SCENE = 5


def _apply_tpw_correction_current(rad, offset, trans):
    return (rad - offset * rad) / trans


def _classify_post_correction(T7c, T14c, Tbc7, Tbc14, day_pixel, sc, alb_ij, is_cloudy):
    offset_day = (BT7_MIN_SOLAR_COEF * sc) if day_pixel else 0.0

    min_t14 = 285.0 if day_pixel else 265.0  
    min_t7 = (285.0 + offset_day) if day_pixel else 265.0
    
    if T14c < min_t14 or T7c < min_t7:
        return FailChar.F3
    if T14c - Tbc14 < 0.25:
        if ((not np.isnan(alb_ij) and alb_ij > 0.15) or is_cloudy) and T7c - Tbc7 > 10.0:
            return FailChar.F10
        return FailChar.F4
    if T7c - Tbc7 < 2.0:
        return FailChar.F5
    return FailChar.NONE


def _evaluate_stage(stage_name, r7, r14, r7b, r14b, coeffs7, coeffs14, day_pixel, sc, alb_ij):
    """Calcula temperaturas y evalúa fail_char para una etapa específica con doble fondo."""
    if r7 <= 0 or r14 <= 0 or r7b <= 0 or r14b <= 0:
        return None, None, None, None, "CONV_ERROR"
    
    t7 = float(planck_temp_from_coeffs(r7, **coeffs7))
    t14 = float(planck_temp_from_coeffs(r14, **coeffs14))
    tb7 = float(planck_temp_from_coeffs(r7b, **coeffs7))
    tb14 = float(planck_temp_from_coeffs(r14b, **coeffs14))
    
    if t7 <= 0 or t14 <= 0 or tb7 <= 0 or tb14 <= 0:
        return None, None, None, None, "CONV_ERROR"
        
    # En etapas intermedias solo evaluamos límite absoluto (F3). Los contrastes van al final.
    if stage_name != "difraccion":
        offset_day = (BT7_MIN_SOLAR_COEF * sc) if day_pixel else 0.0
        min_t14 = 285.0 if day_pixel else 265.0  
        min_t7 = (285.0 + offset_day) if day_pixel else 265.0
        
        if t14 < min_t14 or t7 < min_t7:
            fc = FailChar.F3
        else:
            fc = FailChar.NONE
    else:
        fc = _classify_post_correction(t7, t14, tb7, tb14, day_pixel, sc, alb_ij, is_cloudy=False)
        
    return t7, t14, tb7, tb14, fc


def _diagnose_pixel_chain(r7, r14, r7_bkg_raw, r14_bkg_raw, offset7, trans7,
                          offset14, trans14, em7, em14, coeffs7, coeffs14,
                          pixel_sza, day_pixel, sc, alb_ij):
    
    stages_trace = {}
    
    # 0. Crudo
    t7, t14, tb7, tb14, fc = _evaluate_stage("crudo", r7, r14, r7_bkg_raw, r14_bkg_raw, coeffs7, coeffs14, day_pixel, sc, alb_ij)
    stages_trace["crudo"] = {"T7": t7, "T14": t14, "Tb7": tb7, "Tb14": tb14, "fc": fc}
    
    # 1. TPW
    r7_corr = _apply_tpw_correction_current(r7, offset7, trans7)
    r14_corr = _apply_tpw_correction_current(r14, offset14, trans14)
    r7_bkg_corr = _apply_tpw_correction_current(r7_bkg_raw, offset7, trans7)
    r14_bkg_corr = _apply_tpw_correction_current(r14_bkg_raw, offset14, trans14)
    
    t7, t14, tb7, tb14, fc = _evaluate_stage("TPW", r7_corr, r14_corr, r7_bkg_corr, r14_bkg_corr, coeffs7, coeffs14, day_pixel, sc, alb_ij)
    stages_trace["TPW"] = {"T7": t7, "T14": t14, "Tb7": tb7, "Tb14": tb14, "fc": fc}

    # 2. Emisividad
    r7_em = r7_corr / em7 if r7_corr > 0 else -1
    r14_em = r14_corr / em14 if r14_corr > 0 else -1
    
    t7, t14, tb7, tb14, fc = _evaluate_stage("emisividad", r7_em, r14_em, r7_bkg_corr, r14_bkg_corr, coeffs7, coeffs14, day_pixel, sc, alb_ij)
    stages_trace["emisividad"] = {"T7": t7, "T14": t14, "Tb7": tb7, "Tb14": tb14, "fc": fc}

    # 3. Solar
    try:
        r7_solar, rad7from14_bkg = _solar_correction(r7_em, r7_bkg_corr, r14_bkg_corr, em7, coeffs7, coeffs14, pixel_sza)
    except:
        r7_solar, rad7from14_bkg = -1, -1
        
    t7, t14, tb7, tb14, fc = _evaluate_stage("solar", r7_solar, r14_em, r7_bkg_corr, r14_bkg_corr, coeffs7, coeffs14, day_pixel, sc, alb_ij)
    stages_trace["solar"] = {"T7": t7, "T14": t14, "Tb7": tb7, "Tb14": tb14, "fc": fc}

    r14_corr_em = r14_corr / em14

    # 4. Difracción
    r7_diff = (r7_solar - DIFFRAC_CH7_SUB * rad7from14_bkg) / DIFFRAC_CH7_DIV if r7_solar > 0 else -1
    r14_diff = (r14_corr_em - DIFFRAC_CH14_SUB * r14_bkg_corr) / DIFFRAC_CH14_DIV if r14_corr > 0 else -1

    t7, t14, tb7, tb14, fc = _evaluate_stage("difraccion", r7_diff, r14_diff, r7_bkg_corr, r14_bkg_corr, coeffs7, coeffs14, day_pixel, sc, alb_ij)
    stages_trace["difraccion"] = {"T7": t7, "T14": t14, "Tb7": tb7, "Tb14": tb14, "fc": fc}

    # Encontrar dónde falla por primera vez
    first_fail_stage = "threshold"
    final_fc = stages_trace["difraccion"]["fc"]
    
    for stage in ["crudo", "TPW", "emisividad", "solar", "difraccion"]:
        if stages_trace[stage]["fc"] not in (FailChar.NONE, "CONV_ERROR", None):
            first_fail_stage = stage
            final_fc = stages_trace[stage]["fc"]
            break
        if stages_trace[stage]["fc"] == "CONV_ERROR":
            first_fail_stage = f"CONV_ERROR_en_{stage}"
            final_fc = "ERROR"
            break

    return stages_trace, first_fail_stage, final_fc


def diagnose_scene(ts, stats):
    ref_mask_path = MASK_DIR / f"{ts}.npy"
    if not ref_mask_path.exists():
        return
    ref_mask = np.load(ref_mask_path).astype(np.uint8)

    inp = load_fdca_input(timestamp=ts, region=REGION, dataset_root=str(DATASET_ROOT), config_path=str(CONFIG_PATH), verbose=False)
    FPT = getattr(inp, "FPT", 0.0) or 0.0
    lut_tpw = build_tpw_lut()

    fire_mask_p1, fail_char_p1, candidates = run_part1(
        bt7=inp.bt7, rad7=inp.rad7, bt14=inp.bt14, rad14=inp.rad14,
        bt13=inp.bt13, rad13=inp.rad13, bt15=inp.bt15, refl2=inp.refl2,
        latitudes=inp.latitudes, longitudes=inp.longitudes,
        sza=inp.sza, glint_angle=inp.glint_angle, lza=inp.lza, azimuth=inp.azimuth,
        tpw=inp.tpw, emiss7=inp.emiss7, emiss14=inp.emiss14,
        lut_tpw=lut_tpw, FPT=FPT, coeffs7=inp.coeffs7, coeffs14=inp.coeffs14, coeffs13=inp.coeffs13,
        land_mask=inp.land_mask, eco_mask=eco_mask, data_quality=getattr(inp, "data_quality", None),
    )

    candidate_mask = np.zeros_like(ref_mask, dtype=bool)
    for c in candidates: candidate_mask[c.i, c.j] = True

    L, W = inp.bt7.shape
    albedo, vis_brightness, is_day, sza_cos = calculate_albedo(L, W, inp.sza, inp.refl2)

    use_hybrid = FPT > FPT_THRESHOLD
    if use_hybrid and inp.bt13 is not None and inp.rad13 is not None:
        rad13_in_ch7 = planck_rad_from_coeffs(inp.bt13, **inp.coeffs7)
        rad14_in_ch7 = planck_rad_from_coeffs(inp.bt14, **inp.coeffs7)
        use_ch13 = np.abs(inp.rad7 - rad13_in_ch7) < np.abs(inp.rad7 - rad14_in_ch7)
        bt14_eff = np.where(use_ch13, inp.bt13, inp.bt14)
        rad14_eff = np.where(use_ch13, inp.rad13, inp.rad14)
    else:
        bt14_eff = inp.bt14
        rad14_eff = inp.rad14

    rad14_in_7 = planck_rad_from_coeffs(bt14_eff, **inp.coeffs7)
    refl = inp.rad7 - rad14_in_7
    bad_rad = (inp.rad7 < 0) | (rad14_eff < 0)
    refl = np.where(bad_rad, -9999.0, refl)

    for code, code_name in TARGET_CODES.items():
        fn_coords = np.argwhere((ref_mask == code) & ~candidate_mask)
        if len(fn_coords) == 0: continue
        
        stats[code_name]["total"] += len(fn_coords)

        for idx, (i, j) in enumerate(fn_coords):
            i, j = int(i), int(j)
            day_pixel = bool(is_day[i, j])
            sc = sza_cos[i, j]
            alb_ij = albedo[i, j] if inp.refl2 is not None else np.nan

            p1_mask = fire_mask_p1[i, j]
            p1_fc = fail_char_p1[i, j]
            
            p1_mask_str = p1_mask.name if hasattr(p1_mask, 'name') else str(p1_mask)
            p1_fc_str = p1_fc.name if hasattr(p1_fc, 'name') else str(p1_fc)

            bkg = compute_background(i, j, inp.bt7, bt14_eff, refl, vis_brightness, albedo if inp.refl2 is not None else None, inp.land_mask, sza_cos, is_day)
            if bkg is None:
                stats[code_name]["fail_stage"]["NO_BACKGROUND"] += 1
                continue

            col = _tpw_lut_indices(float(inp.tpw[i, j]), float(inp.lza[i, j]))
            trans7, trans14 = float(lut_tpw[2, col]), float(lut_tpw[3, col])
            offset7, offset14 = float(lut_tpw[4, col]), float(lut_tpw[5, col])

            r7, r14 = float(inp.rad7[i, j]), float(rad14_eff[i, j])
            r7_bkg_raw = planck_rad_from_coeffs(bkg.temp7_bkg_mean, **inp.coeffs7)
            r14_bkg_raw = planck_rad_from_coeffs(bkg.temp14_bkg_mean, **inp.coeffs14)
            
            trace, first_fail, final_fc = _diagnose_pixel_chain(
                r7, r14, r7_bkg_raw, r14_bkg_raw, offset7, trans7, offset14, trans14, 
                float(inp.emiss7[i, j]), float(inp.emiss14[i, j]), inp.coeffs7, inp.coeffs14,
                float(inp.sza[i, j]), day_pixel, sc, alb_ij
            )
            
            if final_fc == FailChar.NONE:
                first_fail = f"rechazado_por_run_part1 (mask: {p1_mask_str}, fc: {p1_fc_str})"

            stats[code_name]["fail_stage"][first_fail] += 1
            if final_fc is not None:
                fc_val = final_fc.value if hasattr(final_fc, 'value') else final_fc
                stats[code_name]["final_fc"][fc_val] += 1

            if idx < MAX_PIXELS_PER_SCENE:
                print(f"\n({i},{j}) {code_name} - SZA: {inp.sza[i,j]:.1f}° | p1_mask: {p1_mask_str} | p1_fc: {p1_fc_str}")
                print(f"{'Etapa':<12} {'T7':>8} {'T14':>8} {'Tb7':>8} {'Tb14':>8} {'ΔT14':>8} {'fail_char'}")
                for stage, data in trace.items():
                    if data['fc'] is None: continue
                    dt14 = data['T14'] - data['Tb14'] if data['T14'] and data['Tb14'] else 0
                    fc_str = data['fc'].name if hasattr(data['fc'], 'name') else str(data['fc'])
                    print(f"{stage:<12} {data['T7']:8.2f} {data['T14']:8.2f} {data['Tb7']:8.2f} {data['Tb14']:8.2f} {dt14:8.2f}  {fc_str}")


stats = {
    "LOW_PROB": {"total": 0, "fail_stage": defaultdict(int), "final_fc": defaultdict(int)},
    "TEMP_LOW": {"total": 0, "fail_stage": defaultdict(int), "final_fc": defaultdict(int)}
}

for ts in DEBUG_TIMESTAMPS:
    diagnose_scene(ts, stats)

print("\n" + "="*50)
print("RESUMEN DE RECALL BAJO")
print("="*50)

for cat, data in stats.items():
    if data["total"] == 0: continue
    print(f"\n{cat} — {data['total']} FN")
    print("\nPerdidos por primera vez en:")
    for stage, count in data["fail_stage"].items():
        print(f"  {stage:<45} {count}")
    
    print("\nDistribución de fail_char final:")
    for fc, count in sorted(data["final_fc"].items(), key=lambda x: str(x[0])):
        print(f"  fail_char={fc:<5} {count}")