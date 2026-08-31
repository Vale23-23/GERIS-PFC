"""
Auditoría de performance del FDCA implementado contra la máscara final de NOAA
(ABI-L2-FDCF "Mask"), sobre una muestra de N timestamps aleatorios.

Corre Parte I + Parte II igual que run_fdca.py, pero además captura TODOS los
resultados intermedios de cada píxel (el gate donde murió, los thresholds
contextuales que se calcularon, las BT corregidas, Dozier, y las tres
condiciones de eliminación de Parte II) y los cruza con la máscara de NOAA
para responder: ¿por dónde se pierde recall y por dónde se pierde precisión?

Uso
---
  python -m fdca.run_audit --n 5 --seed 42
  python -m fdca.run_audit --n 5 --only-with-fire
  python -m fdca.run_audit --timestamps 20251117_1820,20251207_0550
  python -m fdca.run_audit --n 3 --dataset-root data --temporal-source none

Salidas en results/audit/<run_id>/
---------------------------------
  report.md              informe legible (funnel de recall, desglose de FP, tablas)
  summary.json           métricas agregadas + por escena, en JSON
  pixels.csv             UNA FILA POR PÍXEL DE INTERÉS con todos los intermedios
                         (píxeles de fuego en NOAA, píxeles de fuego nuestros y
                         todos los candidatos de Parte I) → tabla para afinar
  part2_trace.csv        una fila por candidato con los intermedios de Parte II
  <timestamp>/arrays.npz máscaras + stage por escena, para graficar después

Métricas (todas por píxel, no por "count" de focos)
---------------------------------------------------
  * total fuego/no-fuego (sin distinguir etiqueta): precision, recall, F1
  * por etiqueta exacta   (10-15 y 30-35)
  * por etiqueta base     (30-35 colapsados a 10-15; independiente del filtro
                           temporal, que depende del historial disponible)
  * Parte I sola: recall techo de los candidatos
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from fdca.algorithm import _to_epoch
from fdca.constants import (
    BKG_MAX_ITER,
    BT7_MIN_NIGHT,
    BT7_MIN_SOLAR_COEF,
    BT7_REFL_THRESH_NIGHT,
    BT7_REFL_THRESH_SOLAR,
    FailChar,
)
from fdca.dataset import (
    default_dataset_root,
    ensure_timestamp_data,
    missing_required_files,
)
from fdca.fdca_adapter import load_fdca_input
from fdca.part1 import Stage, run_part1
from fdca.part2 import run_part2

# ── Códigos de fuego (Tabla 3.11 del ATBD) ───────────────────────────────────
FIRE_CODES = (10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35)
BASE_FIRE_CODES = (10, 11, 12, 13, 14, 15)
CODE_LABELS = {
    10: "procesado (Dozier OK)",
    11: "saturado",
    12: "nube/humo",
    13: "alta probabilidad",
    14: "media probabilidad",
    15: "baja probabilidad",
}
REF_DIR = "ABI-L2-FDCF-Mask"


def to_base_code(mask: np.ndarray) -> np.ndarray:
    """Colapsa 30-35 (fuego + historial temporal) a 10-15."""
    out = np.asarray(mask).copy()
    temporal = np.isin(out, (30, 31, 32, 33, 34, 35))
    out[temporal] -= 20
    return out


# ── Métricas ─────────────────────────────────────────────────────────────────
def scores(reference: np.ndarray, prediction: np.ndarray) -> dict:
    """Conteos y precision/recall/F1 por píxel para dos máscaras booleanas."""
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.count_nonzero(reference & prediction))
    fp = int(np.count_nonzero(~reference & prediction))
    fn = int(np.count_nonzero(reference & ~prediction))
    tn = int(np.count_nonzero(~reference & ~prediction))
    return finish_scores({"tp": tp, "fp": fp, "fn": fn, "tn": tn})


def finish_scores(counts: dict) -> dict:
    """Agrega precision/recall/F1 a un dict con tp/fp/fn/tn (permite micro-suma)."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = dict(counts)
    out.update({
        "support_ref": tp + fn,
        "support_pred": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })
    return out


def add_scores(total: dict | None, new: dict) -> dict:
    """Suma micro de conteos de dos dicts de métricas."""
    if total is None:
        return {k: new[k] for k in ("tp", "fp", "fn", "tn")}
    return {k: total[k] + new[k] for k in ("tp", "fp", "fn", "tn")}


# ── Descubrimiento de timestamps ─────────────────────────────────────────────
def discover_timestamps(dataset_root: str, region: str) -> list[str]:
    """Timestamps con máscara de referencia NOAA e inputs mínimos presentes."""
    ref_dir = Path(dataset_root) / region / REF_DIR
    if not ref_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de máscaras de referencia: {ref_dir}\n"
            "Sin la máscara de NOAA no hay contra qué auditar."
        )
    found = []
    for path in sorted(ref_dir.glob("*.npy")):
        timestamp = path.stem
        if not missing_required_files(timestamp, region, dataset_root):
            found.append(timestamp)
    return found


def load_reference(dataset_root: str, region: str, timestamp: str) -> np.ndarray:
    """Máscara final de NOAA como int32.

    Los .npy se guardaron como int8, así que los códigos > 127 (150, 153, 200,
    201, 240...) quedaron con overflow; se recuperan reinterpretando a uint8.
    """
    path = Path(dataset_root) / region / REF_DIR / f"{timestamp}.npy"
    return np.load(path).astype(np.uint8).astype(np.int32)


def count_reference_fires(dataset_root: str, region: str, timestamp: str) -> int:
    return int(np.count_nonzero(np.isin(
        load_reference(dataset_root, region, timestamp), FIRE_CODES)))


def pick_timestamps(args) -> list[str]:
    """Muestra aleatoria (reproducible por --seed) de timestamps disponibles."""
    if args.timestamps:
        return [t.strip() for t in args.timestamps.split(",") if t.strip()]

    available = discover_timestamps(args.dataset_root, args.region)
    if not available:
        raise SystemExit(
            f"No hay escenas completas con máscara de referencia en "
            f"{args.dataset_root}/{args.region}"
        )

    pool = available
    if args.only_with_fire:
        pool = [t for t in available
                if count_reference_fires(args.dataset_root, args.region, t) > 0]
        if not pool:
            raise SystemExit(
                "Ninguna escena disponible tiene píxeles de fuego en la "
                "referencia de NOAA; corré sin --only-with-fire."
            )

    n = min(args.n, len(pool))
    if n < args.n:
        print(f"Aviso: se pidieron {args.n} timestamps y sólo hay {len(pool)} "
              f"disponibles{' con fuego' if args.only_with_fire else ''}.")
    rng = random.Random(args.seed)
    return sorted(rng.sample(pool, n))


# ── Estado temporal (ATBD 3.4.2.16) ──────────────────────────────────────────
def build_temporal_state(source: str, timestamp: str, region: str,
                         dataset_root: str, shape: tuple, own_state: np.ndarray | None,
                         download: bool) -> np.ndarray | None:
    """
    Devuelve prev_fire_mask (segundos desde 2001 del último fuego por píxel).

    source:
      "reference" → historial de las últimas 12 h tomado de las máscaras de
                    NOAA (es lo que hace run_fdca.py hoy).  Ojo: usa el
                    producto de referencia como entrada, así que el offset +20
                    no es una predicción independiente; sirve para comparar
                    etiquetas de categoría sin arrastrar el error de arranque.
      "own"       → historial construido con nuestras propias detecciones de
                    las escenas ya procesadas en esta corrida (sin fuga, pero
                    casi siempre vacío si la muestra es aleatoria y dispersa).
      "none"      → filtro temporal desactivado (ningún código llega a 30-35).
    """
    if source == "none":
        return None
    if source == "own":
        return own_state
    from fdca.temporal_filter import TemporalFilter

    def _download(ts: str) -> None:
        ensure_timestamp_data(timestamp=ts, region=region,
                              dataset_root=dataset_root, download=True)

    temporal = TemporalFilter(
        data_root=dataset_root,
        region=region,
        timestamp=timestamp,
        shape=shape,
        lookback_hours=12,
        download_callback=_download if download else None,
    )
    return temporal.load_previous_fires()


# ── Corrida de una escena ────────────────────────────────────────────────────
def run_scene(timestamp: str, args, own_state: np.ndarray | None) -> dict:
    """Corre Parte I + Parte II sobre una escena y arma toda la auditoría."""
    ensure_timestamp_data(timestamp=timestamp, region=args.region,
                          dataset_root=args.dataset_root, download=args.download)

    inp = load_fdca_input(timestamp=timestamp, region=args.region,
                          dataset_root=args.dataset_root,
                          config_path=args.config, verbose=False)

    reference = load_reference(args.dataset_root, args.region, timestamp)
    if reference.shape != inp.bt7.shape:
        raise ValueError(
            f"{timestamp}: la referencia tiene shape {reference.shape} y la "
            f"escena {inp.bt7.shape}"
        )

    prev_fire_mask = build_temporal_state(
        args.temporal_source, timestamp, args.region, args.dataset_root,
        inp.bt7.shape, own_state, args.download,
    )

    # ── Parte I, con captura de intermedios ──────────────────────────────────
    diag: dict = {}
    started = datetime.now()
    fire_mask_p1, fail_char_p1, candidates = run_part1(
        bt7=inp.bt7, rad7=inp.rad7,
        bt14=inp.bt14, rad14=inp.rad14,
        bt13=inp.bt13, rad13=inp.rad13,
        bt15=inp.bt15, refl2=inp.refl2,
        latitudes=inp.latitudes, longitudes=inp.longitudes,
        sza=inp.sza, glint_angle=inp.glint_angle,
        lza=inp.lza, azimuth=inp.azimuth,
        tpw=inp.tpw, emiss7=inp.emiss7, emiss14=inp.emiss14,
        lut_tpw=inp.lut_tpw, FPT=inp.FPT,
        coeffs7=inp.coeffs7, coeffs14=inp.coeffs14, coeffs13=inp.coeffs13,
        land_mask=inp.land_mask, eco_mask=inp.eco_mask,
        data_quality=inp.data_quality,
        diag_out=diag,
    )
    part1_seconds = (datetime.now() - started).total_seconds()

    # ── Parte II, con traza por candidato ────────────────────────────────────
    started = datetime.now()
    part2_trace: list[dict] = []
    fire_mask_p2, fail_char_p2, confirmed = run_part2(
        candidates=candidates,
        fire_mask=fire_mask_p1.copy(),
        fail_char_arr=fail_char_p1.copy(),
        prev_fire_mask=prev_fire_mask,
        current_epoch=_to_epoch(inp.scan_time),
        trace_out=part2_trace,
        detection_policy=args.detection_policy,
    )
    part2_seconds = (datetime.now() - started).total_seconds()

    candidate_mask = np.zeros(reference.shape, dtype=bool)
    for cand in candidates:
        candidate_mask[cand.i, cand.j] = True
    confirmed_mask = np.zeros(reference.shape, dtype=bool)
    for cand in confirmed:
        confirmed_mask[cand.i, cand.j] = True

    scene = {
        "timestamp": timestamp,
        "region": args.region,
        "shape": list(reference.shape),
        "n_candidates": len(candidates),
        "n_confirmed": len(confirmed),
        "part1_seconds": part1_seconds,
        "part2_seconds": part2_seconds,
        "temporal_source": args.temporal_source,
        "n_prev_fire_pixels": (0 if prev_fire_mask is None
                               else int(np.count_nonzero(prev_fire_mask > 0))),
        "metrics": scene_metrics(reference, candidate_mask, fire_mask_p2),
        "funnel": recall_funnel(reference, fire_mask_p1, fail_char_p1, diag["stage"]),
        "fp_breakdown": fp_breakdown(reference, fire_mask_p2, part2_trace),
        "part2_paths": part2_path_counts(part2_trace),
        "bkg_approach": bkg_approach_summary(diag, candidate_mask),
    }
    arrays = {
        "reference": reference,
        "fire_mask_p1": fire_mask_p1,
        "fire_mask_p2": fire_mask_p2,
        "fail_char_p1": fail_char_p1,
        "fail_char_p2": fail_char_p2,
        "stage": diag["stage"],
        "candidate_mask": candidate_mask,
        "confirmed_mask": confirmed_mask,
        "bkg_hist_won": diag["bkg_hist_won"],
        "half_width": diag["half_width"],
    }
    pixel_rows = build_pixel_rows(timestamp, inp, reference, fire_mask_p1,
                                  fire_mask_p2, fail_char_p1, fail_char_p2,
                                  candidate_mask, confirmed_mask, diag,
                                  part2_trace, args.all_pixels)
    trace_rows = [dict(row, timestamp=timestamp) for row in part2_trace]
    return {"scene": scene, "arrays": arrays,
            "pixel_rows": pixel_rows, "trace_rows": trace_rows}


# ── Métricas de una escena ───────────────────────────────────────────────────
def scene_metrics(reference: np.ndarray, candidate_mask: np.ndarray,
                  fire_mask_p2: np.ndarray) -> dict:
    """Binario total, por etiqueta exacta y por etiqueta base (todo por píxel)."""
    reference_fire = np.isin(reference, FIRE_CODES)
    predicted_fire = np.isin(fire_mask_p2, FIRE_CODES)
    reference_base = to_base_code(reference)
    predicted_base = to_base_code(fire_mask_p2)

    return {
        # Parte I: techo de recall — un fuego que Parte I descarta ya no se
        # puede recuperar en Parte II.
        "part1_binary": scores(reference_fire, candidate_mask),
        "part2_binary": scores(reference_fire, predicted_fire),
        "by_code": {
            str(code): scores(reference == code, fire_mask_p2 == code)
            for code in FIRE_CODES
        },
        "by_base_code": {
            str(code): scores(reference_base == code, predicted_base == code)
            for code in BASE_FIRE_CODES
        },
        "confusion": confusion_counts(reference, fire_mask_p2),
    }


def confusion_counts(reference: np.ndarray, prediction: np.ndarray) -> dict:
    """Códigos NOAA vs códigos nuestros, sólo donde alguno de los dos es fuego."""
    interesting = np.isin(reference, FIRE_CODES) | np.isin(prediction, FIRE_CODES)
    pairs = Counter(zip(reference[interesting].tolist(),
                        prediction[interesting].tolist()))
    out: dict = {}
    for (ref_code, pred_code), count in sorted(pairs.items()):
        out.setdefault(str(ref_code), {})[str(pred_code)] = count
    return out


def recall_funnel(reference: np.ndarray, fire_mask_p1: np.ndarray,
                  fail_char_p1: np.ndarray, stage: np.ndarray) -> dict:
    """
    Dónde mueren los píxeles que NOAA marcó como fuego.

    Es la respuesta directa a "¿por dónde se pierde recall?": para cada gate
    de Parte I, cuántos píxeles de fuego de la referencia llegaron hasta ahí y
    ahí se cayeron, con el código de máscara y el FailChar con que quedaron.
    """
    reference_fire = np.isin(reference, FIRE_CODES)
    total = int(np.count_nonzero(reference_fire))
    per_stage: dict = {}
    if total:
        stages = stage[reference_fire]
        for value, count in sorted(Counter(stages.tolist()).items()):
            selected = reference_fire & (stage == value)
            per_stage[str(int(value))] = {
                "gate": Stage.killed_by(value),
                "n_reference_fire": int(count),
                "fire_mask_codes": {
                    str(int(k)): int(v) for k, v in
                    sorted(Counter(fire_mask_p1[selected].tolist()).items())},
                "fail_chars": {
                    str(int(k)): int(v) for k, v in
                    sorted(Counter(fail_char_p1[selected].tolist()).items())},
            }
    return {
        "n_reference_fire": total,
        "by_stage": per_stage,
        # Distribución de stage en TODA la escena: contexto para saber cuánto
        # filtra cada gate en general (no sólo sobre los fuegos).
        "all_pixels_by_stage": {
            str(int(k)): int(v) for k, v in
            sorted(Counter(stage.ravel().tolist()).items())},
    }


def fp_breakdown(reference: np.ndarray, fire_mask_p2: np.ndarray,
                 part2_trace: list[dict]) -> dict:
    """Falsos positivos: qué etiqueta les pusimos y qué dice NOAA en ese píxel."""
    reference_fire = np.isin(reference, FIRE_CODES)
    predicted_fire = np.isin(fire_mask_p2, FIRE_CODES)
    false_positive = predicted_fire & ~reference_fire
    trace_by_pixel = {(row["i"], row["j"]): row for row in part2_trace}

    fail_char_out: Counter = Counter()
    for (i, j) in zip(*np.nonzero(false_positive)):
        row = trace_by_pixel.get((int(i), int(j)))
        if row is not None:
            fail_char_out[row.get("fail_char_out", "?")] += 1

    return {
        "n_false_positive": int(np.count_nonzero(false_positive)),
        "predicted_codes": {
            str(int(k)): int(v) for k, v in
            sorted(Counter(fire_mask_p2[false_positive].tolist()).items())},
        "reference_codes_at_fp": {
            str(int(k)): int(v) for k, v in
            sorted(Counter(reference[false_positive].tolist()).items())},
        "fail_char_out": {str(k): int(v) for k, v in sorted(fail_char_out.items())},
    }


def bkg_approach_summary(diag: dict, candidate_mask: np.ndarray) -> dict:
    """
    Resumen de la elección estadístico vs histograma (ATBD 3.4.2.5).

    Se reporta sobre los píxeles que llegaron a tener background (los únicos
    donde la comparación existe) y aparte sobre los candidatos de Parte I, que
    son los que terminan importando para la máscara final.
    """
    hist_won = diag["bkg_hist_won"]
    has_bkg = hist_won >= 0

    def _block(selected: np.ndarray) -> dict:
        n = int(np.count_nonzero(selected))
        if not n:
            return {"n": 0}
        stat_std = diag["t7_stat_std"][selected]
        hist_std = diag["t7_hist_std"][selected]
        return {
            "n": n,
            "n_stat": int(np.count_nonzero(hist_won[selected] == 0)),
            "n_hist": int(np.count_nonzero(hist_won[selected] == 1)),
            "median_t7_stat_std": float(np.nanmedian(stat_std)),
            "median_t7_hist_std": float(np.nanmedian(hist_std)),
            "median_n_hist_selected": float(np.median(
                diag["n_hist_selected"][selected])),
            "half_width_counts": {
                str(int(k)): int(v) for k, v in
                sorted(Counter(diag["half_width"][selected].tolist()).items())},
        }

    return {"with_background": _block(has_bkg),
            "candidates": _block(has_bkg & candidate_mask)}


def part2_path_counts(part2_trace: list[dict]) -> dict:
    """Cuántos candidatos elimina cada condición de 3.4.2.14 y cómo se reetiquetan."""
    eliminated = [r for r in part2_trace if r.get("eliminated")]
    survived = [r for r in part2_trace if not r.get("eliminated")]
    confirmed = [r for r in survived if not r.get("policy_rejected")]
    return {
        "n_candidates": len(part2_trace),
        "n_eliminated": len(eliminated),
        "n_confirmed": len(confirmed),
        "policy_rejected": len(survived) - len(confirmed),
        "eliminated_by_reason": {
            k: v for k, v in
            sorted(Counter(r.get("elim_reason", "?") for r in eliminated).items())},
        "reassign_glint_edge": sum(1 for r in survived if r.get("reassign_glint_edge")),
        "reassign_fog_edge": sum(1 for r in survived if r.get("reassign_fog_edge")),
        "upgraded": {
            str(k): v for k, v in
            sorted(Counter(r.get("upgraded_code") for r in survived).items(),
                   key=lambda kv: str(kv[0]))},
        "temporally_filtered": sum(1 for r in survived if r.get("temporally_filtered")),
        "final_codes": {
            str(k): v for k, v in
            sorted(Counter(r.get("final_code") for r in survived).items(),
                   key=lambda kv: str(kv[0]))},
    }


# ── Tabla por píxel ──────────────────────────────────────────────────────────
def _get(array, i, j, default=float("nan")):
    """Lectura tolerante: devuelve NaN si la banda opcional no está."""
    if array is None:
        return default
    return float(array[i, j])


def bkg_approach_label(flag) -> str:
    """1 → histograma ganó, 0 → estadístico ganó, -1 → nunca hubo background."""
    value = int(flag)
    return "hist" if value == 1 else "stat" if value == 0 else ""


def build_pixel_rows(timestamp: str, inp, reference: np.ndarray,
                     fire_mask_p1: np.ndarray, fire_mask_p2: np.ndarray,
                     fail_char_p1: np.ndarray, fail_char_p2: np.ndarray,
                     candidate_mask: np.ndarray, confirmed_mask: np.ndarray,
                     diag: dict, part2_trace: list[dict],
                     all_pixels: bool) -> list[dict]:
    """
    Una fila por píxel de interés con TODO lo intermedio.

    "De interés" = fuego en la referencia de NOAA, o fuego para nosotros, o
    candidato de Parte I.  Con --all-pixels se vuelca la escena completa.

    Las columnas `mg_*` son márgenes: valor observado menos el umbral con el
    que se lo comparó.  Márgen >= 0 significa "pasó ese test".  Sirven para
    saber cuánto habría que mover un threshold para recuperar un fuego perdido
    (o para matar un falso positivo).
    """
    stage = diag["stage"]
    bt14_eff = diag["bt14_eff"]
    refl = diag["refl"]
    albedo = diag["albedo"]
    is_day = diag["is_day"]
    sza_cos = diag["sza_cos"]
    reference_fire = np.isin(reference, FIRE_CODES)
    predicted_fire = np.isin(fire_mask_p2, FIRE_CODES)

    if all_pixels:
        selected = np.ones(reference.shape, dtype=bool)
    else:
        selected = reference_fire | predicted_fire | candidate_mask

    trace_by_pixel = {(row["i"], row["j"]): row for row in part2_trace}
    rows: list[dict] = []

    for i, j in zip(*np.nonzero(selected)):
        i, j = int(i), int(j)
        ref_fire = bool(reference_fire[i, j])
        pred_fire = bool(predicted_fire[i, j])
        verdict = ("TP" if ref_fire and pred_fire else
                   "FN" if ref_fire else
                   "FP" if pred_fire else "TN")

        bt7 = _get(inp.bt7, i, j)
        bt14 = _get(bt14_eff, i, j)
        diff_bt = bt7 - bt14
        day = bool(is_day[i, j])
        cos_sza = float(sza_cos[i, j])
        bt7_bkg = _get(diag["bt7_bkg"], i, j)
        bt14_bkg = _get(diag["bt14_bkg"], i, j)
        refl_ij = _get(refl, i, j)
        reflb = _get(diag["reflb"], i, j)
        std_7b14b = _get(diag["std_7b14b"], i, j)
        std_7b = _get(diag["std_7b"], i, j)
        std_reflb = _get(diag["std_reflb"], i, j)
        std_reflb_max = _get(diag["std_reflb_max"], i, j)

        row = {
            # ── identidad ────────────────────────────────────────────────────
            "timestamp": timestamp,
            "i": i, "j": j,
            "lat": _get(inp.latitudes, i, j),
            "lon": _get(inp.longitudes, i, j),
            # ── verdad NOAA vs predicción ────────────────────────────────────
            "ref_code": int(reference[i, j]),
            "ref_fire": int(ref_fire),
            "pred_code": int(fire_mask_p2[i, j]),
            "pred_fire": int(pred_fire),
            "verdict": verdict,
            "ref_base_code": int(to_base_code(reference[i, j:j + 1])[0]),
            "pred_base_code": int(to_base_code(fire_mask_p2[i, j:j + 1])[0]),
            # ── recorrido dentro de Parte I ──────────────────────────────────
            "p1_code": int(fire_mask_p1[i, j]),
            "stage": int(stage[i, j]),
            "stage_killed_by": Stage.killed_by(stage[i, j]),
            "reached_candidate": int(bool(candidate_mask[i, j])),
            "confirmed_part2": int(bool(confirmed_mask[i, j])),
            "fail_char_p1": int(fail_char_p1[i, j]),
            "fail_char_p2": int(fail_char_p2[i, j]),
            # ── inputs ───────────────────────────────────────────────────────
            "bt7": bt7,
            "bt14_eff": bt14,
            "use_ch13": int(bool(diag["use_ch13"][i, j])),
            "bt15": _get(inp.bt15, i, j),
            "refl2": _get(inp.refl2, i, j),
            "albedo": _get(albedo, i, j),
            "sza": _get(inp.sza, i, j),
            "lza": _get(inp.lza, i, j),
            "glint_angle": _get(inp.glint_angle, i, j),
            "tpw": _get(inp.tpw, i, j),
            "emiss7": _get(inp.emiss7, i, j),
            "emiss14": _get(inp.emiss14, i, j),
            "eco_code": int(diag["eco_mask_fixed"][i, j]),
            "is_day": int(day),
            "refl": refl_ij,
            "vis_brightness": _get(diag["vis_brightness"], i, j),
            # ── background (3.4.2.5) ────────────────────────────────────────
            "n_passes": int(diag["n_passes"][i, j]),
            "bt7_bkg": bt7_bkg,
            "bt14_bkg": bt14_bkg,
            "bt7_bkg_std": _get(diag["bt7_bkg_std"], i, j),
            "bt14_bkg_std": _get(diag["bt14_bkg_std"], i, j),
            # ── elección estadístico vs histograma (3.4.2.5) ────────────────
            "bkg_approach": bkg_approach_label(diag["bkg_hist_won"][i, j]),
            "t7_stat_std": _get(diag["t7_stat_std"], i, j),
            "t7_hist_std": _get(diag["t7_hist_std"], i, j),
            "t7_std_gap_stat_minus_hist": (_get(diag["t7_stat_std"], i, j)
                                           - _get(diag["t7_hist_std"], i, j)),
            "n_hist_selected": int(diag["n_hist_selected"][i, j]),
            "half_width": int(diag["half_width"][i, j]),
            "reflb": reflb,
            "rad_diff_sigma": _get(diag["rad_diff_sigma"], i, j),
            "albedo_bkg": _get(diag["alb_bkg"], i, j),
            "is_cloudy": int(diag["is_cloudy"][i, j]),
            "sat_flag": int(diag["sat_flag"][i, j]),
            "pass_along_scan": int(diag["pass_along"][i, j]),
            # ── thresholds contextuales (3.4.2.6) ───────────────────────────
            "std_7b14b": std_7b14b,
            "std_7b": std_7b,
            "std_reflb": std_reflb,
            "std_reflb_max": std_reflb_max,
            "bt7_min_thr": BT7_MIN_NIGHT + (BT7_MIN_SOLAR_COEF * cos_sza if day else 0.0),
            "bt7_refl_thr": (BT7_REFL_THRESH_NIGHT
                             + (BT7_REFL_THRESH_SOLAR * cos_sza if day else 0.0)),
            # ── cantidades comparadas contra esos thresholds ────────────────
            "diff_bt7_bt14": diff_bt,
            "bt7_minus_bkg7": bt7 - bt7_bkg,
            "bt14_minus_bkg14": bt14 - bt14_bkg,
            "refl_minus_reflb": refl_ij - reflb,
            # ── márgenes (>= 0 → pasó el test) ──────────────────────────────
            "mg_absdiff_vs_2K": abs(diff_bt) - 2.0,
            "mg_d714_vs_std_7b14b": diff_bt - std_7b14b,
            "mg_d7bkg_vs_std_7b": (bt7 - bt7_bkg) - std_7b,
            "mg_refl_vs_std_reflb": refl_ij - std_reflb,
            "mg_refl_vs_std_reflb_max": refl_ij - std_reflb_max,
            "mg_reflmreflb_vs_std_reflb_max": (refl_ij - reflb) - std_reflb_max,
            "mg_bt14_vs_bkg14_minus20": (bt14 - bt14_bkg) + 20.0,
            "n_passes_over_max": int(diag["n_passes"][i, j] > BKG_MAX_ITER),
            # ── correcciones (3.4.2.8) y Dozier (3.4.2.10) ─────────────────
            "bt7_corr": _get(diag["bt7_corr"], i, j),
            "bt14_corr": _get(diag["bt14_corr"], i, j),
            "bt7_bkg_corr": _get(diag["bt7_bkg_corr"], i, j),
            "bt14_bkg_corr": _get(diag["bt14_bkg_corr"], i, j),
            "skip_dozier": int(diag["skip_dozier"][i, j]),
            "dozier_valid": int(diag["dozier_valid"][i, j]),
            "fire_temp": _get(diag["fire_temp"], i, j),
            "fire_frac": _get(diag["fire_frac"], i, j),
            "frp": _get(diag["frp"], i, j),
        }
        trace = trace_by_pixel.get((i, j))
        if trace is not None:
            row.update({k: v for k, v in trace.items() if k not in ("i", "j")})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], preferred: list[str]) -> None:
    """CSV con columnas ordenadas: primero las preferidas, después el resto."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = set()
    for row in rows:
        keys.update(row)
    ordered = [k for k in preferred if k in keys]
    ordered += sorted(k for k in keys if k not in ordered)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


PIXEL_COLUMNS_FIRST = [
    "timestamp", "i", "j", "lat", "lon",
    "verdict", "ref_code", "pred_code", "ref_fire", "pred_fire",
    "ref_base_code", "pred_base_code",
    "stage", "stage_killed_by", "p1_code", "fail_char_p1", "fail_char_p2",
    "reached_candidate", "confirmed_part2", "eliminated", "elim_reason",
    "bkg_approach", "t7_stat_std", "t7_hist_std",
    "t7_std_gap_stat_minus_hist", "n_hist_selected", "half_width",
]
TRACE_COLUMNS_FIRST = [
    "timestamp", "i", "j", "fail_char_in", "eliminated", "elim_reason",
    "p2_cond1", "p2_cond2", "p2_cond3", "p2_refl_ok",
    "base_code", "upgraded_code", "final_code", "temporally_filtered",
]


# ── Agregación entre escenas (micro: se suman TP/FP/FN/TN) ────────────────────
def aggregate(scenes: list[dict]) -> dict:
    part1 = part2 = None
    by_code: dict = {str(c): None for c in FIRE_CODES}
    by_base: dict = {str(c): None for c in BASE_FIRE_CODES}
    funnel: dict = {}
    funnel_all: Counter = Counter()
    n_reference_fire = 0
    fp_pred: Counter = Counter()
    fp_ref: Counter = Counter()
    paths: Counter = Counter()
    elim_reasons: Counter = Counter()
    final_codes: Counter = Counter()
    bkg_counts: dict = {"with_background": Counter(), "candidates": Counter()}
    half_widths: dict = {"with_background": Counter(), "candidates": Counter()}

    for scene in scenes:
        metrics = scene["metrics"]
        part1 = add_scores(part1, metrics["part1_binary"])
        part2 = add_scores(part2, metrics["part2_binary"])
        for code in by_code:
            by_code[code] = add_scores(by_code[code], metrics["by_code"][code])
        for code in by_base:
            by_base[code] = add_scores(by_base[code], metrics["by_base_code"][code])

        n_reference_fire += scene["funnel"]["n_reference_fire"]
        for stage_value, info in scene["funnel"]["by_stage"].items():
            entry = funnel.setdefault(stage_value, {"gate": info["gate"], "n": 0})
            entry["n"] += info["n_reference_fire"]
        for stage_value, count in scene["funnel"]["all_pixels_by_stage"].items():
            funnel_all[stage_value] += count

        fp_pred.update({k: v for k, v in scene["fp_breakdown"]["predicted_codes"].items()})
        fp_ref.update({k: v for k, v in scene["fp_breakdown"]["reference_codes_at_fp"].items()})
        for key in ("n_candidates", "n_eliminated", "n_confirmed",
                    "reassign_glint_edge", "reassign_fog_edge",
                    "temporally_filtered", "policy_rejected"):
            paths[key] += scene["part2_paths"][key]
        elim_reasons.update(scene["part2_paths"]["eliminated_by_reason"])
        final_codes.update(scene["part2_paths"]["final_codes"])

        for group in ("with_background", "candidates"):
            block = scene["bkg_approach"][group]
            if not block["n"]:
                continue
            bkg_counts[group].update({k: block[k] for k in ("n", "n_stat", "n_hist")})
            half_widths[group].update(block["half_width_counts"])

    return {
        "n_scenes": len(scenes),
        "n_reference_fire_pixels": n_reference_fire,
        "part1_binary": finish_scores(part1) if part1 else None,
        "part2_binary": finish_scores(part2) if part2 else None,
        "by_code": {k: finish_scores(v) for k, v in by_code.items() if v},
        "by_base_code": {k: finish_scores(v) for k, v in by_base.items() if v},
        "recall_funnel": dict(sorted(funnel.items(), key=lambda kv: int(kv[0]))),
        "all_pixels_by_stage": dict(sorted(funnel_all.items(), key=lambda kv: int(kv[0]))),
        "fp_predicted_codes": dict(sorted(fp_pred.items(), key=lambda kv: int(kv[0]))),
        "fp_reference_codes": dict(sorted(fp_ref.items(), key=lambda kv: int(kv[0]))),
        "part2_paths": dict(paths),
        "part2_eliminated_by_reason": dict(sorted(elim_reasons.items())),
        "part2_final_codes": dict(sorted(final_codes.items(), key=lambda kv: str(kv[0]))),
        "bkg_approach": {
            group: dict(
                bkg_counts[group],
                half_width_counts=dict(sorted(half_widths[group].items(),
                                              key=lambda kv: int(kv[0]))),
            )
            for group in ("with_background", "candidates")
        },
    }


MARGIN_COLUMNS = [
    "mg_absdiff_vs_2K",
    "mg_d714_vs_std_7b14b",
    "mg_d7bkg_vs_std_7b",
    "mg_refl_vs_std_reflb",
    "mg_refl_vs_std_reflb_max",
    "mg_reflmreflb_vs_std_reflb_max",
    "mg_bt14_vs_bkg14_minus20",
]


def fn_margin_table(pixel_rows: list[dict]) -> list[dict]:
    """
    Para los fuegos que perdimos (FN), cuán cerca estuvieron del umbral que los
    mató.  La mediana del margen dice cuánto habría que aflojar ese threshold.
    """
    groups: dict = {}
    for row in pixel_rows:
        if row["verdict"] != "FN":
            continue
        groups.setdefault((row["stage"], row["stage_killed_by"]), []).append(row)

    table = []
    for (stage_value, gate), rows in sorted(groups.items()):
        entry = {"stage": stage_value, "gate": gate, "n": len(rows)}
        for column in MARGIN_COLUMNS:
            values = [r[column] for r in rows
                      if isinstance(r.get(column), float) and not np.isnan(r[column])]
            entry[column] = float(np.median(values)) if values else None
        table.append(entry)
    return table


# ── Informe ──────────────────────────────────────────────────────────────────
def _pct(value: float) -> str:
    return f"{value:.2%}"


def _score_row(label: str, s: dict) -> str:
    return (f"| {label} | {_pct(s['precision'])} | {_pct(s['recall'])} | "
            f"{_pct(s['f1'])} | {s['tp']} | {s['fp']} | {s['fn']} | "
            f"{s['support_ref']} |")


SCORE_HEADER = ("| | precision | recall | F1 | TP | FP | FN | píxeles fuego NOAA |\n"
                "|---|---|---|---|---|---|---|---|")


def build_report(args, timestamps: list[str], scenes: list[dict],
                 totals: dict, pixel_rows: list[dict]) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Auditoría FDCA contra la máscara final de NOAA (ABI-L2-FDCF)\n")
    add(f"- Región: `{args.region}`")
    add(f"- Escenas auditadas: {len(timestamps)} → `{', '.join(timestamps)}`")
    add(f"- Semilla de muestreo: `{args.seed}`"
        + ("" if not args.timestamps else " (timestamps explícitos, sin muestreo)"))
    add(f"- Fuente del filtro temporal: `{args.temporal_source}`")
    add(f"- Política de detección: `{args.detection_policy}`")
    if args.detection_policy == "conservative":
        add("  - Perfil empírico fuera de la especificación ATBD: suprime código 12 "
            "y exige ΔBT7 observado ≥ 6 K para código 15; se usa sólo para "
            "cuantificar el trade-off precisión/recall contra NOAA.")
        add("  - Ojo: el historial de 12 h sale de las máscaras de NOAA, así que"
            " el offset +20 (códigos 30-35) no es una predicción independiente."
            " Mirá las tablas por *etiqueta base* para una comparación limpia.")
    add(f"- Píxeles de fuego en la referencia: {totals['n_reference_fire_pixels']}")
    add("- Todas las métricas son **por píxel**, no por foco/cluster.\n")

    add("## 1. Total fuego / no-fuego (sin distinguir etiqueta)\n")
    add(SCORE_HEADER)
    add(_score_row("Parte I (candidatos, techo de recall)", totals["part1_binary"]))
    add(_score_row("Parte II (máscara final)", totals["part2_binary"]))
    add("")
    part1, part2 = totals["part1_binary"], totals["part2_binary"]
    add(f"Parte I propone {part1['support_pred']} píxeles y Parte II confirma "
        f"{part2['support_pred']}. De los {part1['tp']} fuegos reales que Parte I "
        f"encuentra, Parte II conserva {part2['tp']} "
        f"(pierde {part1['tp'] - part2['tp']}).\n")

    add("## 2. Por etiqueta exacta (códigos 10-15 y 30-35)\n")
    add(SCORE_HEADER)
    for code in FIRE_CODES:
        s = totals["by_code"][str(code)]
        if s["support_ref"] or s["support_pred"]:
            add(_score_row(f"código {code}", s))
    add("")

    add("## 3. Por etiqueta base (30-35 colapsados a 10-15)\n")
    add(SCORE_HEADER)
    for code in BASE_FIRE_CODES:
        s = totals["by_base_code"][str(code)]
        if s["support_ref"] or s["support_pred"]:
            add(_score_row(f"{code} — {CODE_LABELS[code]}", s))
    add("")

    add("## 4. ¿Dónde se pierde el recall? Funnel de Parte I\n")
    add("Para cada píxel que NOAA marcó como fuego, el último gate que logró"
        " pasar. El gate que lo mató es el de la columna `gate`.\n")
    add("| stage | gate que lo rechazó | píxeles fuego NOAA | % del total |")
    add("|---|---|---|---|")
    total_fire = max(totals["n_reference_fire_pixels"], 1)
    for stage_value, info in totals["recall_funnel"].items():
        share = info["n"] / total_fire
        marker = " ✅" if int(stage_value) == Stage.CANDIDATE else ""
        add(f"| {stage_value} | {info['gate']}{marker} | {info['n']} | {_pct(share)} |")
    add("")

    margins = fn_margin_table(pixel_rows)
    if margins:
        add("### Márgenes de los fuegos perdidos (FN)\n")
        add("Mediana de `valor observado − umbral` en el gate donde murieron."
            " Negativo = faltó eso para pasar. Es la magnitud en la que habría"
            " que aflojar ese threshold para recuperarlos.\n")
        columns = MARGIN_COLUMNS
        add("| stage | gate | n | " + " | ".join(c.replace("mg_", "") for c in columns) + " |")
        add("|---|---|---|" + "---|" * len(columns))
        for entry in margins:
            cells = ["" if entry[c] is None else f"{entry[c]:+.2f}" for c in columns]
            add(f"| {entry['stage']} | {entry['gate']} | {entry['n']} | "
                + " | ".join(cells) + " |")
        add("")

    add("## 5. ¿Dónde se pierde la precisión? Falsos positivos\n")
    add(f"Total de FP: {totals['part2_binary']['fp']}\n")
    add("| etiqueta que le pusimos | píxeles FP |")
    add("|---|---|")
    for code, count in totals["fp_predicted_codes"].items():
        add(f"| {code} | {count} |")
    add("")
    add("| código NOAA en ese píxel | píxeles FP |")
    add("|---|---|")
    for code, count in totals["fp_reference_codes"].items():
        add(f"| {code} | {count} |")
    add("")

    add("## 6. Recorrido de Parte II (3.4.2.14 - 3.4.2.16)\n")
    paths = totals["part2_paths"]
    add(f"- Candidatos evaluados: {paths['n_candidates']}")
    add(f"- Eliminados como falsa alarma: {paths['n_eliminated']}")
    for reason, count in totals["part2_eliminated_by_reason"].items():
        add(f"  - `{reason}`: {count}")
    add(f"- Confirmados: {paths['n_confirmed']}")
    add(f"- Rechazados por política empírica: {paths['policy_rejected']}")
    add(f"- Reasignados a F11 por borde de nube/glint: {paths['reassign_glint_edge']}")
    add(f"- Reasignados a F11 por borde de niebla: {paths['reassign_fog_edge']}")
    add(f"- Con historial temporal (+20): {paths['temporally_filtered']}")
    add("- Códigos finales asignados: "
        + ", ".join(f"{k}={v}" for k, v in totals["part2_final_codes"].items()))
    add("")

    add("## 7. Background: ¿ganó el enfoque estadístico o el histograma? (3.4.2.5)\n")
    add("Gana el enfoque con menor desvío estándar de BT7; el empate va para"
        " `stat` (`t7_stat_std <= t7_hist_std`).\n")
    add("| grupo | píxeles con background | ganó stat | ganó hist | % hist |")
    add("|---|---|---|---|---|")
    labels = {"with_background": "todos los píxeles con background",
              "candidates": "candidatos de Parte I"}
    for group, label in labels.items():
        block = totals["bkg_approach"][group]
        n = block.get("n", 0)
        if not n:
            continue
        add(f"| {label} | {n} | {block['n_stat']} | {block['n_hist']} | "
            f"{_pct(block['n_hist'] / n)} |")
    add("")
    for group, label in labels.items():
        counts = totals["bkg_approach"][group].get("half_width_counts") or {}
        if counts:
            add(f"- `half_width` final ({label}): "
                + ", ".join(f"{k}→{v}" for k, v in counts.items()))
    add("")
    add("Por escena, las medianas de `t7_stat_std`, `t7_hist_std` y de la"
        " cantidad de píxeles seleccionados por el histograma están en"
        " `summary.json` (`scenes[].bkg_approach`). Por píxel, en `pixels.csv`:"
        " `bkg_approach`, `t7_stat_std`, `t7_hist_std`,"
        " `t7_std_gap_stat_minus_hist`, `n_hist_selected`, `half_width`.\n")

    add("## 8. Por escena\n")
    add("| timestamp | fuego NOAA | candidatos P1 | confirmados P2 | "
        "P recall | P precision | F1 |")
    add("|---|---|---|---|---|---|---|")
    for scene in scenes:
        binary = scene["metrics"]["part2_binary"]
        add(f"| {scene['timestamp']} | {binary['support_ref']} | "
            f"{scene['n_candidates']} | {scene['n_confirmed']} | "
            f"{_pct(binary['recall'])} | {_pct(binary['precision'])} | "
            f"{_pct(binary['f1'])} |")
    add("")

    add("## 9. Archivos generados\n")
    add("- `pixels.csv` — un registro por píxel de interés con inputs,"
        " background, thresholds contextuales, márgenes, BT corregidas, Dozier"
        " y la traza de Parte II. Filtrá por `verdict == 'FN'` para atacar"
        " recall y por `verdict == 'FP'` para atacar precisión.")
    add("- `part2_trace.csv` — un registro por candidato con las tres"
        " condiciones de 3.4.2.14 y los umbrales de confianza de 3.4.2.15.")
    add("- `summary.json` — todas las métricas y desgloses en JSON.")
    add("- `<timestamp>/arrays.npz` — máscaras, `stage`, `bkg_hist_won`"
        " (1 = ganó histograma, 0 = ganó estadístico, -1 = sin background) y"
        " `half_width` por escena.")
    return "\n".join(lines)


def print_console_summary(totals: dict) -> None:
    part1, part2 = totals["part1_binary"], totals["part2_binary"]
    print("\n" + "=" * 72)
    print(f"AUDITORÍA FDCA — {totals['n_scenes']} escenas, "
          f"{totals['n_reference_fire_pixels']} píxeles de fuego en NOAA")
    print("=" * 72)
    print(f"Parte I  (candidatos) : precision={_pct(part1['precision'])} "
          f"recall={_pct(part1['recall'])} f1={_pct(part1['f1'])} "
          f"TP={part1['tp']} FP={part1['fp']} FN={part1['fn']}")
    print(f"Parte II (final)      : precision={_pct(part2['precision'])} "
          f"recall={_pct(part2['recall'])} f1={_pct(part2['f1'])} "
          f"TP={part2['tp']} FP={part2['fp']} FN={part2['fn']}")
    print("\nPor etiqueta base (30-35 colapsados a 10-15):")
    print(f"  {'código':>8} {'precision':>10} {'recall':>10} {'f1':>10} "
          f"{'TP':>5} {'FP':>6} {'FN':>5}")
    for code in BASE_FIRE_CODES:
        s = totals["by_base_code"][str(code)]
        if not (s["support_ref"] or s["support_pred"]):
            continue
        print(f"  {code:>8} {s['precision']:>9.2%} {s['recall']:>9.2%} "
              f"{s['f1']:>9.2%} {s['tp']:>5} {s['fp']:>6} {s['fn']:>5}")
    print("\nDónde mueren los fuegos de NOAA (funnel de Parte I):")
    for stage_value, info in totals["recall_funnel"].items():
        print(f"  stage {stage_value:>3} → {info['n']:>5}  {info['gate']}")
    bkg = totals["bkg_approach"]["with_background"]
    if bkg.get("n"):
        print(f"\nBackground 3.4.2.5: ganó stat en {bkg['n_stat']} píxeles y "
              f"hist en {bkg['n_hist']} ({_pct(bkg['n_hist'] / bkg['n'])} hist)")
    print("=" * 72)


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audita Parte I + Parte II del FDCA contra la máscara de NOAA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=5,
                        help="cantidad de timestamps aleatorios a auditar")
    parser.add_argument("--seed", type=int, default=42,
                        help="semilla del muestreo (reproducible)")
    parser.add_argument("--timestamps", default=None,
                        help="lista explícita separada por comas (ignora --n/--seed)")
    parser.add_argument("--only-with-fire", action="store_true",
                        help="muestrear sólo escenas con fuego en la referencia")
    parser.add_argument("--region", default="uruguay")
    parser.add_argument("--dataset-root", default=default_dataset_root())
    parser.add_argument("--config",
                        default=str(Path(__file__).resolve().parent / "config.yaml"))
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="disable automatic download of missing scenes from Hugging Face")
    parser.set_defaults(download=True)
    parser.add_argument("--temporal-source", default="none",
                        choices=("reference", "own", "none"),
                        help="de dónde sale el historial de 12 h del filtro temporal")
    parser.add_argument("--detection-policy", default="atbd",
                        choices=("atbd", "conservative"),
                        help="atbd = categorías normativas; conservative = "
                             "perfil empírico para subir precisión contra NOAA")
    parser.add_argument("--output-dir", default="results/audit")
    parser.add_argument("--run-id", default=None,
                        help="nombre de la subcarpeta de salida")
    parser.add_argument("--all-pixels", action="store_true",
                        help="volcar en pixels.csv la escena completa, no sólo "
                             "los píxeles de interés (archivos grandes)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    timestamps = pick_timestamps(args)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"AUDITORÍA FDCA | región={args.region} | escenas={len(timestamps)}")
    print(f"Timestamps: {', '.join(timestamps)}")
    print(f"Salida: {out_dir}")
    print("=" * 72)

    scenes: list[dict] = []
    pixel_rows: list[dict] = []
    trace_rows: list[dict] = []
    own_state: np.ndarray | None = None

    for index, timestamp in enumerate(timestamps, start=1):
        print(f"\n[{index}/{len(timestamps)}] {timestamp} ...")
        result = run_scene(timestamp, args, own_state)
        scene, arrays = result["scene"], result["arrays"]

        # Historial propio para la escena siguiente (sólo con --temporal-source own)
        if args.temporal_source == "own":
            if own_state is None:
                own_state = np.zeros(arrays["reference"].shape, dtype=np.int64)
            epoch = int(_to_epoch(datetime.strptime(timestamp, "%Y%m%d_%H%M")))
            own_state[np.isin(arrays["fire_mask_p2"], FIRE_CODES)] = epoch

        scene_dir = out_dir / timestamp
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scene_dir / "arrays.npz", **arrays)

        binary = scene["metrics"]["part2_binary"]
        print(f"  candidatos P1={scene['n_candidates']} "
              f"confirmados P2={scene['n_confirmed']} | "
              f"fuego NOAA={binary['support_ref']} | "
              f"recall={_pct(binary['recall'])} "
              f"precision={_pct(binary['precision'])} "
              f"f1={_pct(binary['f1'])} "
              f"({scene['part1_seconds']:.0f}s + {scene['part2_seconds']:.1f}s)")

        scenes.append(scene)
        pixel_rows.extend(result["pixel_rows"])
        trace_rows.extend(result["trace_rows"])

    totals = aggregate(scenes)
    write_csv(out_dir / "pixels.csv", pixel_rows, PIXEL_COLUMNS_FIRST)
    write_csv(out_dir / "part2_trace.csv", trace_rows, TRACE_COLUMNS_FIRST)

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "region": args.region,
            "dataset_root": str(args.dataset_root),
            "timestamps": timestamps,
            "seed": args.seed,
            "only_with_fire": args.only_with_fire,
            "temporal_source": args.temporal_source,
            "detection_policy": args.detection_policy,
        },
        "totals": totals,
        "fn_margins": fn_margin_table(pixel_rows),
        "scenes": scenes,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    report = build_report(args, timestamps, scenes, totals, pixel_rows)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    print_console_summary(totals)
    print(f"\nInforme  : {out_dir / 'report.md'}")
    print(f"Píxeles  : {out_dir / 'pixels.csv'} ({len(pixel_rows)} filas)")
    print(f"Parte II : {out_dir / 'part2_trace.csv'} ({len(trace_rows)} filas)")
    print(f"JSON     : {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
