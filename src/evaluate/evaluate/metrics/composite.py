from typing import Optional, Dict
from config import DEFAULT_WEIGHTS

def compute_composite_score(
    qwk: float,
    material_f1: float,
    explanation_score: float,
    format_valid_rate: float,
    conflict_f1: float,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Composite Score = weighted average dari metrik utama.
    Digunakan untuk perbandingan model, monitoring, dan early stopping.
    """
    w = weights or DEFAULT_WEIGHTS
    score = (
        w["qwk"]               * qwk
        + w["material_f1"]      * material_f1
        + w["explanation_score"]* explanation_score
        + w["format_valid_rate"]* format_valid_rate
        + w["conflict_f1"]      * conflict_f1
    )
    
    return round(score, 4)