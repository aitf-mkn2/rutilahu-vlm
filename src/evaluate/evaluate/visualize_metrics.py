import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import streamlit as st

# =========================
# 3. KONFLIK
# =========================
def plot_konflik(title, labels, values):
    fig, ax = plt.subplots(figsize=(4,3))

    ax.barh(labels, values)

    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=10)

    st.pyplot(fig)

def generate_visualizations(metrics, output_dir):

    output_dir = Path(output_dir) / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    comp = metrics["component_metrics"]
    konflik = metrics["konflik_metrics"]
    summary = metrics["summary"]

    # =========================
    # HELPER
    # =========================
    def add_labels(values):
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)

    # =========================
    # 1. MATERIAL — MACRO F1
    # =========================
    labels = ["Atap", "Dinding", "Lantai"]
    values = [
        comp["atap"]["material"]["macro_f1"],
        comp["dinding"]["material"]["macro_f1"],
        comp["lantai"]["material"]["macro_f1"],
    ]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, values)
    plt.title("Material — Macro F1")
    plt.ylabel("Score")
    plt.ylim(0, 1)

    add_labels(values)

    plt.tight_layout()
    plt.savefig(output_dir / "material_macro_f1.png")
    plt.close()

    # =========================
    # 2. KONDISI — QWK
    # =========================
    values = [
        comp["atap"]["kondisi"]["qwk"],
        comp["dinding"]["kondisi"]["qwk"],
        comp["lantai"]["kondisi"]["qwk"],
    ]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, values)
    plt.title("Kondisi — QWK")
    plt.ylabel("QWK Score")
    plt.ylim(0, 1)

    add_labels(values)

    plt.tight_layout()
    plt.savefig(output_dir / "kondisi_qwk.png")
    plt.close()
    
    # =========================
    # 4. SUMMARY — HORIZONTAL BAR 
    # =========================
    labels = ["QWK", "Material F1", "Explanation", "Format", "Conflict"]
    values = [
        summary["avg_qwk"],
        summary["avg_material_f1"],
        summary["avg_explanation"],
        metrics["meta"]["format_valid_rate"],
        summary["conflict_f1"],
    ]

    plt.figure(figsize=(6, 4))
    plt.barh(labels, values)

    for i, v in enumerate(values):
        plt.text(v + 0.02, i, f"{v:.2f}", va='center')

    plt.xlim(0, 1)
    plt.title("Overall Summary")

    plt.tight_layout()
    plt.savefig(output_dir / "summary.png")
    plt.close()
    
    # =========================
    # 5. EXPLANATION — PER ASPECT 🔥
    # =========================
    # COVERAGE
    exp = metrics["explanation_metrics"]["per_aspect"]

    coverage_keys = [k for k in exp if "coverage" in k]
    values = [exp[k] for k in coverage_keys]

    labels = ["Atap", "Dinding", "Lantai"]

    plt.figure(figsize=(4,4))
    plt.pie(values, labels=labels, autopct='%1.0f%%')
    plt.title("Coverage (Komponen disebut di penjelasan)")

    plt.savefig(output_dir / "exp_coverage.png")
    plt.close()
    
    # CONSISTENCY (horizontal bar)
    cons_keys = [k for k in exp if "consistency" in k]

    labels = [k.replace("_consistency", "").replace("_", " ") for k in cons_keys]
    values = [exp[k] for k in cons_keys]

    # sort biar enak dibaca
    pairs = sorted(zip(labels, values), key=lambda x: x[1])
    labels, values = zip(*pairs)

    plt.figure(figsize=(6,4))
    plt.barh(labels, values)

    for i, v in enumerate(values):
        plt.text(v + 0.02, i, f"{v:.2f}", va='center')

    plt.xlim(0, 1)
    plt.title("Consistency (Material & Kondisi)")

    plt.tight_layout()
    plt.savefig(output_dir / "exp_consistency.png")
    plt.close()
    
    # FORMAT & RULE CHECK 
    rule_keys = ["opening_format", "min_length", "conflict_explanation"]

    labels = [k.replace("_", " ") for k in rule_keys if k in exp]
    values = [exp[k] for k in rule_keys if k in exp]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values)

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.xticks(rotation=20)  
    plt.ylim(0, 1)
    plt.title("Explanation Rules")

    plt.tight_layout()
    plt.savefig(output_dir / "exp_rules.png")
    plt.close()