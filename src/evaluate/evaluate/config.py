from dataclasses import dataclass, field

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARNING] scikit-learn tidak ditemukan. Install dengan: pip install scikit-learn")

# Urutan ordinal untuk kondisi (digunakan pada QWK)
KONDISI_ORDER = ["baik", "rusak_ringan", "rusak_sedang", "rusak_berat", "tidak_terlihat"]
KONDISI_MAP: dict[str, int] = {k: i for i, k in enumerate(KONDISI_ORDER)}

COMPONENTS = ["atap", "dinding", "lantai"]


VALID_MATERIAL: dict[str, list[str]] = {
    "atap": [
        "beton", "genteng", "seng", "asbes", "kayu", "sirap",
        "jerami", "ijuk", "daun_daunan", "rumbia", "tidak_terlihat", "lainnya",
    ],
    "dinding": [
        "tembok", "plesteran_anyaman_bambu", "kawat", "kayu", "papan",
        "gypsum", "grc", "calciboard", "anyaman_bambu", "batang_kayu",
        "bambu", "tidak_terlihat","lainnya",
    ],
    "lantai": [
        "marmer", "granit", "keramik", "parket", "vinil", "karpet",
        "ubin", "tegel", "teraso", "kayu", "papan", "semen",
        "bata_merah", "bambu", "tanah","tidak_terlihat","lainnya",
    ],
}   

VALID_KONDISI = {
    "baik",
    "rusak_ringan",
    "rusak_sedang",
    "rusak_berat",
    "tidak_terlihat",
}

NORMALIZE_KONDISI = {
    "baik": "baik",
    "bagus": "baik",
    "kerusakan ringan": "rusak_ringan",
    "rusak ringan": "rusak_ringan",
    "rusak sedang": "rusak_sedang",
    "rusak berat": "rusak_berat",
    "tidak terlihat": "tidak_terlihat",
}

DEFAULT_WEIGHTS = {
    "qwk":              0.35,
    "material_f1":      0.25,
    "explanation_score":0.20,
    "format_valid_rate":0.10,
    "conflict_f1":      0.10,
}

def normalize_mat(x: str | None) -> str | None:
    if not x:
        return x
    return x.lower().replace(" ", "_").strip()


@dataclass
class EvaluationResult:
    """Agregasi hasil evaluasi."""
    format_valid_rate: float = 0.0
    classification: dict = field(default_factory=dict)
    konflik: dict = field(default_factory=dict)
    explanation: dict = field(default_factory=dict)
    composite_score: float = 0.0

    n_samples: int = 0
    n_valid: int = 0