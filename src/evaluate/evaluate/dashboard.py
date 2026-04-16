import streamlit as st
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from visualize_metrics import plot_konflik

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="VLM Evaluation Dashboard", layout="wide")

st.title("VLM Evaluation Dashboard")

# =========================
# LOAD FILE
# =========================
metrics_path = st.text_input(
    "Path ke metrics.json",
    "outputs/metrics/metrics_test.json"
)

if not Path(metrics_path).exists():
    st.error("File tidak ditemukan")
    st.stop()

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

comp = metrics["component_metrics"]
konflik = metrics["konflik_metrics"]
summary = metrics["summary"]
meta = metrics["meta"]

# =========================
# TOP METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Composite Score", f"{metrics['composite_score']:.4f}")
col2.metric("Format Valid", f"{meta['format_valid_rate']:.2%}")
col3.metric("Avg QWK", f"{summary['avg_qwk']:.4f}")
col4.metric("Material F1", f"{summary['avg_material_f1']:.4f}")

st.divider()

# =========================
# HELPER
# =========================
def plot_bar(title, labels, values):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(labels, values)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1)

    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)

    st.pyplot(fig)


def plot_radar(title, labels, values):
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(4,4))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0,1)
    ax.set_title(title, fontsize=10)

    st.pyplot(fig)


def plot_horizontal_bar(title, labels, values):
    fig, ax = plt.subplots(figsize=(5,3))

    ax.barh(labels, values)

    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=8)

    ax.set_xlim(0,1)
    ax.set_title(title, fontsize=10)

    st.pyplot(fig)

# =========================
# ROW 1 (MATERIAL + KONDISI)
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Material (Macro F1)")
    labels = ["Atap", "Dinding", "Lantai"]
    values = [
        comp["atap"]["material"]["macro_f1"],
        comp["dinding"]["material"]["macro_f1"],
        comp["lantai"]["material"]["macro_f1"],
    ]
    plot_bar("Material Macro F1", labels, values)

with col2:
    st.subheader("Kondisi (QWK)")
    values = [
        comp["atap"]["kondisi"]["qwk"],
        comp["dinding"]["kondisi"]["qwk"],
        comp["lantai"]["kondisi"]["qwk"],
    ]
    plot_bar("Kondisi QWK", labels, values)

st.divider()

# =========================
# ROW 2 (KONFLIK + SUMMARY)
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Konflik (Binary)")
    labels_k = ["Precision", "Recall", "F1"]
    values_k = [
        konflik["precision"],
        konflik["recall"],
        konflik["f1"],
    ]
    plot_konflik("Konflik Metrics", labels_k, values_k)

with col2:
    st.subheader("📈 Summary")
    labels_s = ["QWK", "Material F1", "Explanation", "Format", "Conflict"]
    values_s = [
        summary["avg_qwk"],
        summary["avg_material_f1"],
        summary["avg_explanation"],
        meta["format_valid_rate"],
        summary["conflict_f1"],
    ]
    plot_horizontal_bar("Overall Summary", labels_s, values_s)
    
# =========================
# EXPLANATION VISUAL
# =========================

st.divider()
st.subheader("Explanation Quality")

exp = metrics.get("explanation_metrics", {}).get("per_aspect", {})

if not exp:
    st.warning("Explanation metrics tidak tersedia")
else:

    col1, col2 = st.columns(2)

    # =========================
    # COVERAGE (PIE)
    # =========================
    with col1:
        st.subheader("**Coverage (Komponen disebut)**")

        coverage_keys = [k for k in exp if "coverage" in k]
        values = [exp[k] for k in coverage_keys]

        labels = ["Atap", "Dinding", "Lantai"]

        fig, ax = plt.subplots(figsize=(3,3))
        ax.pie(values, labels=labels, autopct='%1.0f%%')
        ax.set_title("Coverage", fontsize=10)

        st.pyplot(fig)

    # =========================
    # CONSISTENCY (BARH)
    # =========================
    with col2:
        st.subheader("**Consistency (Material & Kondisi)**")

        cons_keys = [k for k in exp if "consistency" in k]

        labels = [k.replace("_consistency", "").replace("_", " ") for k in cons_keys]
        values = [exp[k] for k in cons_keys]

        pairs = sorted(zip(labels, values), key=lambda x: x[1])
        labels, values = zip(*pairs)

        fig, ax = plt.subplots(figsize=(5,3))
        ax.barh(labels, values)

        for i, v in enumerate(values):
            ax.text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=8)

        ax.set_xlim(0,1)
        ax.set_title("Consistency", fontsize=10)

        st.pyplot(fig)

    # =========================
    # RULE CHECK (BOTTOM)
    # =========================
    st.subheader("**Rules Check**")

    rule_keys = ["opening_format", "min_length", "conflict_explanation"]

    labels = [k.replace("_", " ") for k in rule_keys if k in exp]
    values = [exp[k] for k in rule_keys if k in exp]

    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(labels, values)

    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)

    ax.set_ylim(0,1)
    ax.set_title("Explanation Rules", fontsize=10)

    st.pyplot(fig)
