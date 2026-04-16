from typing import Any
import numpy as np

from parser import ParsedOutput
from config import COMPONENTS, KONDISI_ORDER, KONDISI_MAP
from metrics.konflik import compute_konflik_metrics

# Optional sklearn
try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
    )

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# SECTION 4 — CLASSIFICATION METRICS


def _safe_f1(y_true: list, y_pred: list, **kwargs) -> float:
    if not HAS_SKLEARN or not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, zero_division=0, **kwargs))


def _safe_accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    if HAS_SKLEARN:
        return float(accuracy_score(y_true, y_pred))
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def compute_material_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """
    Metrik untuk prediksi material (klasifikasi multi-kelas, nominal).
    Metrik utama: Macro F1-score.
    """
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "per_class_f1": {}, "n_samples": 0}

    accuracy = _safe_accuracy(y_true, y_pred)
    macro_f1 = _safe_f1(y_true, y_pred, average="macro")
    labels = sorted(set(y_true) | set(y_pred))

    per_class: dict[str, float] = {}
    if HAS_SKLEARN:
        scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        per_class = {lbl: float(s) for lbl, s in zip(labels, scores)}
        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    else:
        cm = []

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {k: round(v, 4) for k, v in per_class.items()},
        "confusion_matrix": cm,
        "labels": labels,
        "n_samples": len(y_true),
    }


def compute_qwk(y_true: list[str], y_pred: list[str]) -> float:
    """
    Quadratic Weighted Kappa (QWK) untuk label ordinal kondisi.
    Penalti proporsional dengan kuadrat jarak antar label.
    Implementasi manual agar tidak bergantung full pada sklearn.
    """
    if not y_true:
        return 0.0

    # Map ke integer ordinal; label tak dikenal → index terakhir
    def to_int(lbl: str) -> int:
        return KONDISI_MAP.get(lbl.lower(), len(KONDISI_ORDER) - 1)

    n_cls = len(KONDISI_ORDER)
    yt_int = [to_int(v) for v in y_true]
    yp_int = [to_int(v) for v in y_pred]
    n = len(yt_int)

    # Observed matrix O
    O = np.zeros((n_cls, n_cls), dtype=float)
    for t, p in zip(yt_int, yp_int):
        O[t][p] += 1

    # Expected matrix E
    hist_t = np.bincount(yt_int, minlength=n_cls) / n
    hist_p = np.bincount(yp_int, minlength=n_cls) / n
    E = np.outer(hist_t, hist_p) * n

    # Weight matrix W (quadratic)
    W = np.array(
        [[(i - j) ** 2 / (n_cls - 1) ** 2 for j in range(n_cls)] for i in range(n_cls)]
    )

    num = np.sum(W * O)
    denom = np.sum(W * E)
    if denom == 0:
        return 1.0
    return round(float(1 - num / denom), 4)


def compute_kondisi_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """
    Metrik untuk prediksi kondisi (klasifikasi ordinal).
    Metrik utama: QWK.
    """
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "qwk": 0.0, "n_samples": 0}

    return {
        "accuracy": round(_safe_accuracy(y_true, y_pred), 4),
        "macro_f1": round(_safe_f1(y_true, y_pred, average="macro"), 4),
        "qwk": compute_qwk(y_true, y_pred),
        "n_samples": len(y_true),
    }


def compute_classification_metrics(
    valid_pairs: list[tuple[ParsedOutput, ParsedOutput, str]],
) -> tuple[dict, dict]:
    """
    Wrapper untuk menghitung:
    - metrics per komponen
    - konflik metrics
    """

    comp_data = {
        comp: {"mat_true": [], "mat_pred": [], "kond_true": [], "kond_pred": []}
        for comp in COMPONENTS
    }

    konflik_true, konflik_pred = [], []

    for pp, pr, _ in valid_pairs:
        for comp in COMPONENTS:
            comp_data[comp]["mat_true"].append(
                getattr(pr, f"{comp}_material") or "lainnya"
            )
            comp_data[comp]["mat_pred"].append(
                getattr(pp, f"{comp}_material") or "lainnya"
            )

            comp_data[comp]["kond_true"].append(
                getattr(pr, f"{comp}_kondisi") or "tidak_terlihat"
            )
            comp_data[comp]["kond_pred"].append(
                getattr(pp, f"{comp}_kondisi") or "tidak_terlihat"
            )

        if pr.konflik_dinding is not None and pp.konflik_dinding is not None:
            konflik_true.append(pr.konflik_dinding)
            konflik_pred.append(pp.konflik_dinding)

    component_metrics = {}
    for comp in COMPONENTS:
        d = comp_data[comp]
        component_metrics[comp] = {
            "material": compute_material_metrics(d["mat_true"], d["mat_pred"]),
            "kondisi": compute_kondisi_metrics(d["kond_true"], d["kond_pred"]),
        }

    konflik_metrics = compute_konflik_metrics(konflik_true, konflik_pred)

    return component_metrics, konflik_metrics
