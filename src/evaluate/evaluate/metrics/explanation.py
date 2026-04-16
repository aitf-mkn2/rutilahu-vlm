from collections import defaultdict
import numpy as np
from typing import Any

from parser import ParsedOutput
from config import normalize_mat

# SECTION 5 — EXPLANATION QUALITY (RULE-BASED)

def evaluate_explanation(
    parsed_pred: ParsedOutput, parsed_ref: ParsedOutput
) -> dict[str, Any]:
    """
    Evaluasi kualitas explanation menggunakan rule-based checks.
    Setiap aspek menghasilkan skor 0 atau 1.
    Skor akhir = rata-rata dari seluruh aspek (rentang 0–1).
    """
    scores: dict[str, int] = {}
    penj = (parsed_pred.penjelasan or "").lower()

    # 1. Opening Format: harus diawali "rumah ini"
    scores["opening_format"] = int(penj.startswith("rumah ini"))

    # Panjang minimum explanation
    scores["min_length"] = int(len(penj.split()) > 20)

    # 2. Coverage: menyebut ketiga komponen
    scores["coverage_atap"] = int("atap" in penj)
    scores["coverage_dinding"] = int("dinding" in penj)
    scores["coverage_lantai"] = int("lantai" in penj)

    # 3. Konsistensi Material — material di penjelasan sesuai label prediksi
    def mat_in_penj(material: str | None) -> int:
        if not material:
            return 0
        return int(normalize_mat(material).replace("_", " ") in penj.replace("_", " "))

    scores["material_atap_consistency"] = mat_in_penj(parsed_pred.atap_material)
    scores["material_dinding_consistency"] = mat_in_penj(parsed_pred.dinding_material)
    scores["material_lantai_consistency"] = mat_in_penj(parsed_pred.lantai_material)

    if parsed_ref:
        scores["material_ref_in_explanation"] = int(
            all(
                [
                    normalize_mat(parsed_ref.atap_material or "").replace("_", " ")
                    in penj.replace("_", " "),
                    normalize_mat(parsed_ref.dinding_material or "").replace("_", " ")
                    in penj,
                    normalize_mat(parsed_ref.lantai_material or "").replace("_", " ")
                    in penj,
                ]
            )
        )
    else:
        scores["material_ref_in_explanation"] = 0

    # 4. Konsistensi Kondisi — kata kunci kondisi muncul di penjelasan
    KONDISI_KEYWORDS: dict[str, list[str]] = {
        "baik": ["baik", "terawat", "tidak rusak", "bagus"],
        "rusak_ringan": ["rusak ringan", "kerusakan ringan", "sedikit rusak"],
        "rusak_sedang": ["rusak sedang", "kerusakan sedang", "cukup rusak"],
        "rusak_berat": ["rusak berat", "kerusakan berat", "parah", "rusak parah"],
        "tidak_terlihat": [
            "tidak terlihat",
            "tidak dapat diamati",
            "tidak tampak",
            "tidak terlihat",
        ],
    }

    def kondisi_in_penj(kondisi: str | None) -> int:
        if not kondisi:
            return 0
        keywords = KONDISI_KEYWORDS.get(
            kondisi.lower(), [kondisi.lower().replace("_", " ")]
        )
        return int(any(kw in penj for kw in keywords))

    scores["kondisi_atap_consistency"] = kondisi_in_penj(parsed_pred.atap_kondisi)
    scores["kondisi_dinding_consistency"] = kondisi_in_penj(parsed_pred.dinding_kondisi)
    scores["kondisi_lantai_consistency"] = kondisi_in_penj(parsed_pred.lantai_kondisi)

    # 5. Penanganan tidak_terlihat
    def check_tidak_terlihat(kondisi: str | None, component: str) -> int | None:
        """Return 1/0 jika kondisi=tidak_terlihat, None jika tidak relevan."""
        if kondisi and kondisi.lower() == "tidak_terlihat":
            phrases = ["tidak terlihat", "tidak dapat diamati", "tidak tampak"]
            return int(any(p in penj for p in phrases))
        return None  # tidak dievaluasi

    for comp, attr in [
        ("atap", "atap_kondisi"),
        ("dinding", "dinding_kondisi"),
        ("lantai", "lantai_kondisi"),
    ]:
        val = check_tidak_terlihat(getattr(parsed_pred, attr), comp)
        if val is not None:
            scores[f"tidak_terlihat_{comp}"] = val

    # 6. Penjelasan Konflik (hanya jika konflik=True di prediksi atau referensi)
    if (parsed_pred.konflik_dinding is True) or (
        parsed_ref and parsed_ref.konflik_dinding
    ):
        konflik_phrases = [
            "berbeda",
            "perbedaan",
            "luar",
            "dalam",
            "konflik",
            "tidak sama",
            "eksterior",
            "interior",
        ]
        scores["conflict_explanation"] = int(any(p in penj for p in konflik_phrases))

    # ── Hitung skor akhir ──────────────────────────────────────────────
    valid_scores = {k: v for k, v in scores.items() if v is not None}
    avg_score = (
        round(sum(valid_scores.values()) / len(valid_scores), 4)
        if valid_scores
        else 0.0
    )

    return {
        "score": avg_score,
        "detail": valid_scores,
        "n_aspects": len(valid_scores),
    }


def compute_explanation_metrics(valid_pairs):
    """
    Aggregasi explanation score seluruh dataset
    """

    scores = []
    detail_agg = defaultdict(list)

    for pp, pr, _ in valid_pairs:
        if pp.penjelasan:
            res = evaluate_explanation(pp, pr)
            scores.append(res["score"])

            for k, v in res["detail"].items():
                detail_agg[k].append(v)

    avg_score = round(float(np.mean(scores)), 4) if scores else 0.0

    per_aspect = {k: round(float(np.mean(v)), 4) for k, v in detail_agg.items()}

    return {
        "avg_score": avg_score,
        "per_aspect": per_aspect,
        "n_evaluated": len(scores),
    }
