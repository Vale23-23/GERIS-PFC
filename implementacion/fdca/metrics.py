"""Evaluation metrics against the final FDCA reference mask."""

from __future__ import annotations

import numpy as np


FIRE_CODES = (10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35)


def _binary_scores(reference: np.ndarray, prediction: np.ndarray) -> dict:
    """Return binary fire-detection counts and precision/recall/F1."""
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.count_nonzero(reference & prediction))
    fp = int(np.count_nonzero(~reference & prediction))
    fn = int(np.count_nonzero(reference & ~prediction))
    tn = int(np.count_nonzero(~reference & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def _class_scores(reference: np.ndarray, prediction: np.ndarray, code: int) -> dict:
    """One-vs-rest scores for an exact final fire-mask code."""
    return _binary_scores(reference == code, prediction == code)


def evaluate(reference_mask: np.ndarray, fire_mask_p1: np.ndarray,
             candidate_mask: np.ndarray, fire_mask_p2: np.ndarray) -> dict:
    """Compute Part I and Part II metrics against one reference mask."""
    reference_mask = np.asarray(reference_mask)
    fire_mask_p1 = np.asarray(fire_mask_p1)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    fire_mask_p2 = np.asarray(fire_mask_p2)
    shapes = {reference_mask.shape, fire_mask_p1.shape,
              candidate_mask.shape, fire_mask_p2.shape}
    if len(shapes) != 1:
        raise ValueError(f"Las máscaras tienen shapes incompatibles: {shapes}")

    reference_fire = np.isin(reference_mask, FIRE_CODES)
    prediction_fire_p2 = np.isin(fire_mask_p2, FIRE_CODES)
    recall_by_reference_code = {}
    for code in FIRE_CODES:
        ref_code = reference_mask == code
        recall_by_reference_code[str(code)] = (
            int(np.count_nonzero(ref_code & candidate_mask))
            / int(np.count_nonzero(ref_code))
            if np.count_nonzero(ref_code) else 0.0
        )

    return {
        "fire_codes": list(FIRE_CODES),
        "part1": _binary_scores(reference_fire, candidate_mask),
        "part1_recall_by_reference_code": recall_by_reference_code,
        "part2": _binary_scores(reference_fire, prediction_fire_p2),
        "part2_by_code": {
            str(code): _class_scores(reference_mask, fire_mask_p2, code)
            for code in FIRE_CODES
        },
    }

def _format_scores(scores: dict) -> str:
    return (f"precision={scores['precision']:.2%} | "
            f"recall={scores['recall']:.2%} | f1={scores['f1']:.2%} | "
            f"TP={scores['tp']} FP={scores['fp']} FN={scores['fn']}")


def print_metrics(metrics: dict, reference_path: str) -> None:
    """Print a compact human-readable metrics report."""
    print("\n" + "=" * 72)
    print("MÉTRICAS CONTRA MÁSCARA DE REFERENCIA")
    print(f"Referencia: {reference_path}")
    print("Códigos considerados fuego: " + ", ".join(map(str, FIRE_CODES)))
    print("\nPARTE I — candidatos")
    print("  " + _format_scores(metrics["part1"]))
    print("  Recall Parte I por código de referencia:")
    for code, recall in metrics["part1_recall_by_reference_code"].items():
        print(f"    código {code}: recall={recall:.2%}")

    print("\nPARTE II — máscara final, fuego contra no-fuego")
    print("  " + _format_scores(metrics["part2"]))
    print("\nPARTE II — coincidencia exacta por código")
    print(f"  {'Código':>6} {'Precisión':>12} {'Recall':>12} {'F1':>12} "
          f"{'TP':>6} {'FP':>6} {'FN':>6}")
    for code in FIRE_CODES:
        scores = metrics["part2_by_code"][str(code)]
        print(f"  {code:>6} {scores['precision']:>11.2%} "
              f"{scores['recall']:>11.2%} {scores['f1']:>11.2%} "
              f"{scores['tp']:>6} {scores['fp']:>6} {scores['fn']:>6}")
    print("=" * 72)
