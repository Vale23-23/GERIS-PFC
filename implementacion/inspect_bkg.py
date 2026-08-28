"""
Inspecciona compute_background() para un píxel que se pierde por F2.

Por defecto conserva el caso histórico (39, 157), pero acepta otra escena y
coordenada desde la línea de comandos, por ejemplo:

    python inspect_bkg.py --timestamp 20251117_1820 --i 41 --j 64

Muestra el historial de expansión, las estadísticas de background, los
umbrales contextuales y las dos partes de F2 por separado.
"""

import argparse
import os
from pathlib import Path
import numpy as np

from fdca.constants import *
from fdca.part1 import (
    calculate_albedo,
    _contextual_thresholds,
    _along_scan_reflectivity_test,
)
from fdca.background import compute_background
from fdca.planck import planck_rad_from_coeffs
from fdca.fdca_adapter import load_fdca_input, build_tpw_lut


def _looks_like_repo_root(p: Path) -> bool:
    return (p / "implementacion" / "fdca").is_dir()


def _default_repo_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()
    for candidate in [start, *start.parents]:
        if _looks_like_repo_root(candidate):
            return candidate
    raise FileNotFoundError("Definí GERIS_REPO_ROOT.")


REPO_ROOT = Path(os.environ.get("GERIS_REPO_ROOT", _default_repo_root()))
IMPLEMENTATION_ROOT = REPO_ROOT / "implementacion"
DATASET_ROOT = Path(os.environ.get(
    "GERIS_DATASET_ROOT", str(REPO_ROOT / "implementacion" / "data")
))
CONFIG_PATH = IMPLEMENTATION_ROOT / "fdca" / "config.yaml"
REGION = "uruguay"

# Valores por defecto del caso usado durante el diagnóstico inicial.
DEFAULT_TS = "20251117_1820"
DEFAULT_TARGET_I, DEFAULT_TARGET_J = 39, 157


def main():
    parser = argparse.ArgumentParser(
        description="Inspecciona background y umbrales F2 para un píxel."
    )
    parser.add_argument("--timestamp", default=DEFAULT_TS,
                        help=f"Escena YYYYMMDD_HHMM (default: {DEFAULT_TS})")
    parser.add_argument("--i", type=int, default=DEFAULT_TARGET_I,
                        help=f"Fila del píxel (default: {DEFAULT_TARGET_I})")
    parser.add_argument("--j", type=int, default=DEFAULT_TARGET_J,
                        help=f"Columna del píxel (default: {DEFAULT_TARGET_J})")
    args = parser.parse_args()

    inp = load_fdca_input(timestamp=args.timestamp, region=REGION,
                           dataset_root=str(DATASET_ROOT),
                           config_path=str(CONFIG_PATH), verbose=False)

    L, W = inp.bt7.shape
    albedo, vis_brightness, is_day, sza_cos = calculate_albedo(L, W, inp.sza, inp.refl2)

    use_hybrid = FPT_THRESHOLD is not None and (getattr(inp, "FPT", 0.0) or 0.0) > FPT_THRESHOLD
    use_ch13 = np.zeros(inp.bt7.shape, dtype=bool)
    if (use_hybrid and inp.bt13 is not None and inp.rad13 is not None
            and inp.coeffs13 is not None):
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
    if inp.rad13 is not None:
        bad_rad |= use_ch13 & (inp.rad13 < 0)
    refl = np.where(bad_rad, -9999.0, refl)

    i, j = args.i, args.j
    if not (0 <= i < inp.bt7.shape[0] and 0 <= j < inp.bt7.shape[1]):
        parser.error(
            f"La coordenada ({i},{j}) está fuera de la escena {inp.bt7.shape}."
        )

    day_pixel = bool(is_day[i, j])
    sc = sza_cos[i, j]

    trace = []
    bkg = compute_background(
        i, j, inp.bt7, bt14_eff, refl, vis_brightness,
        albedo if inp.refl2 is not None else None,
        inp.land_mask, sza_cos, is_day, trace=trace,
    )

    print(f"\n{'='*90}\nPíxel ({i},{j}) en {args.timestamp}  —  bt7[i,j]={inp.bt7[i,j]:.2f}K  "
          f"bt14_eff[i,j]={bt14_eff[i,j]:.2f}K  day_pixel={day_pixel}\n{'='*90}")

    print("\n-- Historial de expansión de ventana --")
    for entry in trace:
        if "n_iter" in entry:
            print(f"  intento {entry['n_iter']}: half={entry['half']} "
                  f"window_size={entry['window_size']} n_valid={entry['n_valid']} "
                  f"frac_valid={entry['frac_valid']:.2%} cumple_20%={entry['cumple_20pct']}")

    for entry in trace:
        if entry.get("fase") == "enfoques":
            print("\n-- Comparación de enfoques (ATBD 3.4.2.5) --")
            print(f"  STAT : mean={entry['t7_stat_mean']:.2f}K std={entry['t7_stat_std']:.4f}K")
            print(f"  HIST : mean={entry['t7_hist_mean']:.2f}K std={entry['t7_hist_std']:.4f}K "
                  f"(peak_diff={entry['diff_peak']:.0f}, n_seleccionados={entry['n_hist_selected']})")

            i_lo, i_hi, j_lo, j_hi = entry["valid_bounds"]
            vm = entry["valid_mask"]
            # Posición del propio píxel dentro de la ventana recortada
            local_i, local_j = i - i_lo, j - j_lo
            en_rango = (0 <= local_i < vm.shape[0]) and (0 <= local_j < vm.shape[1])
            autoincluido = bool(vm[local_i, local_j]) if en_rango else False
            print(f"\n  ¿El propio píxel ({i},{j}) está en el rango de la ventana? {en_rango}")
            print(f"  ¿Está marcado como 'valid' (o sea, CONTAMINA su propio background)? {autoincluido}")

            if autoincluido:
                bt7_vals = entry["bt7_vals"]
                n = len(bt7_vals)
                mean_con_self = float(np.mean(bt7_vals))
                # Recalcular excluyendo la instancia con el valor exacto del pixel
                # (aproximado: remueve UNA ocurrencia igual a bt7[i,j])
                bt7_sin_self = bt7_vals[bt7_vals != inp.bt7[i, j]]
                if len(bt7_sin_self) == n:  # no lo encontró exacto (float), remover por índice
                    idx_self = None
                mean_sin_self = float(np.mean(bt7_sin_self)) if len(bt7_sin_self) else np.nan
                print(f"  n={n} pixeles en el background; mean CON autoinclusión={mean_con_self:.3f}K, "
                      f"mean SIN autoinclusión (aprox.)={mean_sin_self:.3f}K "
                      f"(shift={mean_con_self - mean_sin_self:+.3f}K)")

    if bkg is None:
        print("\nNO_BACKGROUND (no convergió)")
        return

    std_7b14b, std_7b, std_reflb, std_reflb_max = _contextual_thresholds(bkg, sc)
    margen = inp.bt7[i, j] - bkg.temp7_bkg_mean
    refl_ij = float(refl[i, j])
    bt7_refl_thr = BT7_REFL_THRESH_NIGHT + (
        BT7_REFL_THRESH_SOLAR * sc if day_pixel else 0.0
    )
    pass_along = _along_scan_reflectivity_test(
        refl, i, j, inp.bt7.shape[1], std_reflb, inp.bt7[i, j], bt7_refl_thr
    )
    f2_temp = margen < std_7b
    f2_refl = (refl_ij < std_reflb_max) or pass_along

    print(f"\n-- Resultado final usado por el algoritmo --")
    print(f"  temp7_bkg_mean elegido = {bkg.temp7_bkg_mean:.3f}K (n_passes={bkg.n_passes}, "
          f"half_width={bkg.half_width}, bkg_count_frac={bkg.bkg_count_frac:.2%})")
    print(f"  bt7[i,j] - temp7_bkg_mean = {margen:+.3f}K")
    print(f"  std_7b (umbral requerido) = {std_7b:.3f}K")
    print(f"  ¿Pasa la parte térmica de F2? {not f2_temp} "
          f"(margen < std_7b: {f2_temp})")
    print(f"  refl_ij = {refl_ij:.6f}")
    print(f"  std_reflb = {std_reflb:.6f}")
    print(f"  std_reflb_max = {std_reflb_max:.6f}")
    print(f"  pass_along = {pass_along}")
    print(f"  ¿Pasa la parte de reflectividad de F2? {not f2_refl} "
          f"(refl < std_reflb_max OR pass_along: {f2_refl})")
    print(f"  ¿Se activa F2 completo? {f2_temp and f2_refl}")
    print(f"  Brecha faltante térmica = {std_7b - margen:.3f}K")

    # ------------------------------------------------------------------
    # NUEVO: Evaluación de falsas alarmas (F4, F5, F10)
    # Aproximamos T7c, T14c y Tbc14 usando las variables disponibles en el inspector
    # (Reemplazar por las variables con corrección de emisividad si las computas aquí)
    T7c = inp.bt7[i, j]
    T14c = bt14_eff[i, j]
    Tbc14 = bkg.temp14_bkg_mean
    alb_ij = albedo[i, j] if inp.refl2 is not None else np.nan
    is_cloudy = False # Reemplazar con la máscara real de nubes si se tiene en el scope

    diff_abs_14 = np.abs(T14c - Tbc14)
    diff_lineal_7_14bkg = T7c - Tbc14

    print(f"\n-- Evaluación de Falsas Alarmas (F4, F5, F10) --")
    print(f"  T7c   (usando bt7)           = {T7c:.3f}K")
    print(f"  T14c  (usando bt14_eff)      = {T14c:.3f}K")
    print(f"  Tbc14 (usando temp14_bkg_mean) = {Tbc14:.3f}K")
    print(f"  Albedo (alb_ij)              = {alb_ij:.3f}")
    
    print(f"\n  Condición 1: |T14c - Tbc14| = {diff_abs_14:.3f}K (Umbral < 0.25K)")
    print(f"  Condición 2: (T7c - Tbc14) = {diff_lineal_7_14bkg:.3f}K")

    # Lógica exacta de tu snippet
    if diff_abs_14 < 0.25:
        cond_nube_albedo = ((not np.isnan(alb_ij) and alb_ij > 0.15) or is_cloudy)
        print(f"  -> ¿Cumple condición de albedo/nubes? {cond_nube_albedo}")
        if cond_nube_albedo and (diff_lineal_7_14bkg > 10.0):
            print("  ❌ RESULTADO: Excluido por FailChar.F10")
        else:
            print("  ❌ RESULTADO: Excluido por FailChar.F4")
            
    elif diff_lineal_7_14bkg < 2.0:
        print("  ❌ RESULTADO: Excluido por FailChar.F5")
        
    else:
        print("  ✅ RESULTADO: Sobrevive a los filtros F4, F5 y F10")

if __name__ == "__main__":
    main()