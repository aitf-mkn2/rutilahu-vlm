from typing import Any, Dict
from config import COMPONENTS


def build_report(metrics: Dict[str, Any], split: str) -> str:
    """Generate human-readable evaluation report."""

    meta    = metrics.get("meta", {})
    comp    = metrics.get("component_metrics", {})
    konflik = metrics.get("konflik_metrics", {})
    exp     = metrics.get("explanation_metrics", {})

    lines: list[str] = []

    def h(title: str) -> None:
        lines.append("")
        lines.append("─" * 50)
        lines.append(f"  {title}")
        lines.append("─" * 50)

    # ── HEADER ─────────────────────────────────────────────
    lines.append("=" * 50)
    lines.append("  VLM EVALUATION REPORT")
    lines.append(f"  Split     : {split.upper()}")
    lines.append(f"  Timestamp : {meta.get('timestamp', '-')}")
    lines.append("=" * 50)

    # ── FORMAT VALIDITY ────────────────────────────────────
    h("FORMAT VALIDITY")
    lines.append(f"  Total Samples   : {meta.get('n_total', 0)}")
    lines.append(f"  Valid Outputs   : {meta.get('n_valid', 0)}")
    lines.append(f"  Invalid Outputs : {meta.get('n_invalid', 0)}")
    lines.append(f"  Format Valid Rate : {meta.get('format_valid_rate', 0):.2%}")
    lines.append(f"  Validity Gap      : {meta.get('validity_gap', 0):.2%}")

    # ── COMPOSITE SCORE ────────────────────────────────────
    h("COMPOSITE SCORE")
    lines.append(f"  ★  {metrics.get('composite_score', 0.0):.4f}")
    lines.append("  Komponen:")
    for k, w in metrics.get("composite_weights", {}).items():
        lines.append(f"    {k:<22} weight={w:.2f}")

    # ── MATERIAL ───────────────────────────────────────────
    h("MATERIAL CLASSIFICATION (per komponen)")
    lines.append(f"  {'Komponen':<10} {'Accuracy':>10} {'Macro F1':>10}")
    lines.append(f"  {'-'*10} {'-'*10} {'-'*10}")

    for c in COMPONENTS:
        m = comp.get(c, {}).get("material", {})
        lines.append(
            f"  {c.capitalize():<10} "
            f"{m.get('accuracy', 0):>10.4f} "
            f"{m.get('macro_f1', 0):>10.4f}"
        )

    lines.append(
        f"  {'AVG':<10} {'':>10} "
        f"{metrics.get('aggregated', {}).get('material_avg_macro_f1', 0):>10.4f}"
    )

    # ── KONDISI ────────────────────────────────────────────
    h("KONDISI CLASSIFICATION — QWK (per komponen)")
    lines.append(f"  {'Komponen':<10} {'Accuracy':>10} {'Macro F1':>10} {'QWK':>10}")
    lines.append(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for c in COMPONENTS:
        k = comp.get(c, {}).get("kondisi", {})
        lines.append(
            f"  {c.capitalize():<10} "
            f"{k.get('accuracy', 0):>10.4f} "
            f"{k.get('macro_f1', 0):>10.4f} "
            f"{k.get('qwk', 0):>10.4f}"
        )

    lines.append(
        f"  {'AVG':<10} {'':>10} {'':>10} "
        f"{metrics.get('aggregated', {}).get('kondisi_avg_qwk', 0):>10.4f}"
    )

    # ── KONFLIK ────────────────────────────────────────────
    h("KONFLIK DINDING (Binary)")
    lines.append(f"  Accuracy  : {konflik.get('accuracy', 0):.4f}")
    lines.append(f"  Precision : {konflik.get('precision', 0):.4f}")
    lines.append(f"  Recall    : {konflik.get('recall', 0):.4f}   ← prioritas utama")
    lines.append(f"  F1        : {konflik.get('f1', 0):.4f}")
    lines.append(f"  N samples : {konflik.get('n_samples', 0)}")

    # ── EXPLANATION ────────────────────────────────────────
    h("EXPLANATION QUALITY")
    lines.append(
        f"  Average Score : {exp.get('avg_score', 0):.4f} "
        f"(N={exp.get('n_evaluated', 0)})"
    )
    lines.append("  Per Aspect:")
    for aspect, score in sorted(exp.get("per_aspect", {}).items()):
        lines.append(f"    {aspect:<38} {score:.4f}")

    # ── INVALID SAMPLES ───────────────────────────────────
    if metrics.get("invalid_samples"):
        h("INVALID SAMPLES (first 10)")
        for s in metrics["invalid_samples"][:10]:
            lines.append(f"    Pred Errors: {s.get('pred_errors', [])}")
            lines.append(f"    Ref Errors : {s.get('ref_errors', [])}")

    lines.append("")
    return "\n".join(lines)