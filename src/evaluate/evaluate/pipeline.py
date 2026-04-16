import json
import logging
import csv
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import Counter
from visualize_metrics import generate_visualizations
import subprocess
import sys
import subprocess



import numpy as np

from config import COMPONENTS, DEFAULT_WEIGHTS
from parser import ParsedOutput, parse_model_output
from loader import (
    load_jsonl,
    align_predictions_references,
    extract_text_from_record,
)

from metrics.classification import compute_classification_metrics
from metrics.explanation import compute_explanation_metrics
from metrics.composite import compute_composite_score
from report import build_report

logger = logging.getLogger(__name__)


def run_evaluation(
    predictions_path: Path,
    references_path: Path,
    output_dir: Path,
    split: str = "test",
) -> dict[str, Any]:

    logger.info("=" * 60)
    logger.info("VLM Evaluation Pipeline")
    logger.info("Predictions : %s", predictions_path)
    logger.info("References  : %s", references_path)
    logger.info("Output dir  : %s", output_dir)
    logger.info("Split       : %s", split)
    logger.info("=" * 60)

    # ── Step 1: Load data ─────────────────────────
    preds_raw = load_jsonl(predictions_path)
    refs_raw  = load_jsonl(references_path)

    pairs = align_predictions_references(preds_raw, refs_raw)

    # ── Step 2: Parse ─────────────────────────────
    parsed_pairs: list[tuple[ParsedOutput, ParsedOutput, str]] = []

    for i, (pred_rec, ref_rec) in enumerate(pairs):
        sample_id = str(i)
        pred_text   = extract_text_from_record(pred_rec) or ""
        ref_text    = extract_text_from_record(ref_rec) or ""

        parsed_pred = parse_model_output(pred_text)
        parsed_ref  = parse_model_output(ref_text)

        parsed_pairs.append((parsed_pred, parsed_ref, sample_id))

    # ── Step 3: Format validity ───────────────────
    total = len(parsed_pairs)
    n_valid = sum(1 for pp, pr, _ in parsed_pairs if pp.is_valid and pr.is_valid)
    format_valid_rate = n_valid / total if total else 0.0

    # ── Step 4: Filter valid ──────────────────────
    valid_pairs = [
        (pp, pr, sid)
        for pp, pr, sid in parsed_pairs
        if pp.is_valid and pr.is_valid
    ]

    invalid_samples = [
        {
            "id": sid,
            "pred_errors": pp.parse_errors,
            "ref_errors": pr.parse_errors,
        }
        for pp, pr, sid in parsed_pairs
        if not pp.is_valid or not pr.is_valid
    ]

    logger.info("Valid pairs   : %d", len(valid_pairs))
    logger.info("Invalid cases : %d", len(invalid_samples))

    # ── Step 5: Classification ────────────────────
    if valid_pairs:
        component_metrics, konflik_metrics = compute_classification_metrics(valid_pairs)
    else:
        component_metrics, konflik_metrics = {}, {"f1": 0.0}

    # ── Step 6: Aggregation ───────────────────────
    if valid_pairs:
        avg_material_f1 = round(
            float(np.mean([component_metrics[c]["material"]["macro_f1"] for c in COMPONENTS])), 4
        )
        avg_qwk = round(
            float(np.mean([component_metrics[c]["kondisi"]["qwk"] for c in COMPONENTS])), 4
        )
    else:
        avg_material_f1 = 0.0
        avg_qwk = 0.0
        

    # ── Step 6.5: Distribution Label ───────────────────────
    material_dist = {}
    kondisi_dist = {}

    for c in COMPONENTS:
        mats = [getattr(pr, f"{c}_material") for _, pr, _ in valid_pairs]
        kons = [getattr(pr, f"{c}_kondisi") for _, pr, _ in valid_pairs]

        material_dist[c] = dict(Counter(mats))
        kondisi_dist[c]  = dict(Counter(kons))


    # ── Step 6.6: Error per component ──────────────────────
    error_per_component = {}

    for c in COMPONENTS:
        total_c = len(valid_pairs)
        correct = sum(
            getattr(pp, f"{c}_material") == getattr(pr, f"{c}_material") and
            getattr(pp, f"{c}_kondisi") == getattr(pr, f"{c}_kondisi")
            for pp, pr, _ in valid_pairs
        )
        error_per_component[c] = round(1 - (correct / total_c), 4) if total_c else 0.0


    # ── Step 6.7: Top error samples ───────────────────────
    top_errors = []

    for pp, pr, sid in parsed_pairs:
        if not pp.is_valid or not pr.is_valid:
            continue

        mismatch = (
            pp.atap_material != pr.atap_material or
            pp.dinding_material != pr.dinding_material or
            pp.lantai_material != pr.lantai_material
        )

        if mismatch:
            top_errors.append({
                "id": sid,
                "pred": pp.__dict__,
                "ref": pr.__dict__,
            })


    # ── Step 6.8: Interpretation ──────────────────────────
    def interpret_score(score):
        if score > 0.8:
            return "sangat baik"
        elif score > 0.6:
            return "baik"
        elif score > 0.4:
            return "cukup"
        else:
            return "buruk"

    # ── Step 7: Explanation ───────────────────────
    exp_result = compute_explanation_metrics(valid_pairs)
    avg_explanation_score = exp_result["avg_score"]

    # ── Step 8: Composite ─────────────────────────
    composite = compute_composite_score(
        qwk=avg_qwk,
        material_f1=avg_material_f1,
        explanation_score=avg_explanation_score,
        format_valid_rate=format_valid_rate,
        conflict_f1=konflik_metrics["f1"],
    )

    # ── Step 9: Metrics dict ──────────────────────
    metrics = {
        "meta": {
            "split": split,
            "timestamp": datetime.now().isoformat(),
            "n_total": total,
            "n_valid": n_valid,
            "n_invalid": total - n_valid,
            "format_valid_rate": round(format_valid_rate, 4),
            "validity_gap": round(1 - format_valid_rate, 4),
            "n_valid_pairs": len(valid_pairs),
        },
        "composite_score": composite,
        "composite_weights": DEFAULT_WEIGHTS,
        "aggregated": {
            "material_avg_macro_f1": avg_material_f1,
            "kondisi_avg_qwk": avg_qwk,
        },
        "summary": {
            "avg_qwk": avg_qwk,
            "avg_material_f1": avg_material_f1,
            "avg_explanation": avg_explanation_score,
            "conflict_f1": konflik_metrics["f1"],
        },
        "component_metrics": component_metrics,
        "konflik_metrics": konflik_metrics,
        "explanation_metrics": {
            "avg_score": avg_explanation_score,
            "per_aspect": exp_result["per_aspect"],
            "n_evaluated": exp_result["n_evaluated"],
        },
        "invalid_samples": invalid_samples,
        "composite_breakdown": {
            "qwk": avg_qwk,
            "material_f1": avg_material_f1,
            "explanation_score": avg_explanation_score,
            "format_valid_rate": format_valid_rate,
            "conflict_f1": konflik_metrics["f1"],
        },
        
        "distribution": {
            "material": material_dist,
            "kondisi": kondisi_dist,
        },

        "error_analysis": {
            "material_error_per_component": error_per_component
        },

        "top_errors": top_errors[:10],

        "interpretation": {
            "composite": interpret_score(composite),
            "qwk": interpret_score(avg_qwk),
            "material": interpret_score(avg_material_f1),
        },
        
        "sample_debug": [
            {
                "id": sid,
                "pred": pp.__dict__,
                "ref": pr.__dict__,
            }
            for pp, pr, sid in parsed_pairs[:5]
        ],
    }

    # ── Step 10: Save ─────────────────────────────
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # JSON metrics
    metrics_path = metrics_dir / f"metrics_{split}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # TXT report
    report_path = metrics_dir / f"report_{split}.txt"
    report_text = build_report(metrics, split)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # CSV summary
    csv_path = metrics_dir / f"summary_{split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["avg_qwk", avg_qwk])
        writer.writerow(["avg_material_f1", avg_material_f1])
        writer.writerow(["explanation_score", avg_explanation_score])
        writer.writerow(["format_valid_rate", format_valid_rate])
        writer.writerow(["conflict_f1", konflik_metrics["f1"]])
        writer.writerow(["composite_score", composite])

    # invalid samples
    with open(metrics_dir / f"invalid_{split}.json", "w", encoding="utf-8") as f:
        json.dump(invalid_samples, f, indent=2, ensure_ascii=False)

    # debug samples
    with open(metrics_dir / f"debug_{split}.json", "w", encoding="utf-8") as f:
        json.dump(metrics["sample_debug"], f, indent=2, ensure_ascii=False)

    logger.info("Saved all outputs to %s", metrics_dir)

    print("\n" + "=" * 60)
    print(report_text)
    print("=" * 60)
    
    # ── SAVE VISUAL DATA ─────────────────────────────
    plot_ready_path = metrics_dir / f"plot_ready_{split}.json"

    with open(plot_ready_path, "w", encoding="utf-8") as f:
        json.dump({
            "labels": COMPONENTS,
            "material_f1": [
                component_metrics.get(c, {}).get("material", {}).get("macro_f1", 0)
                for c in COMPONENTS
            ],
            "qwk": [
                component_metrics.get(c, {}).get("kondisi", {}).get("qwk", 0)
                for c in COMPONENTS
            ],
        }, f, indent=2, ensure_ascii=False)
        
    generate_visualizations(metrics, output_dir)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
        print("🚀 Dashboard launched!")
    except Exception as e:
        print("⚠️ Dashboard gagal:", e)

    return metrics