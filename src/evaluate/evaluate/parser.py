from dataclasses import dataclass, field
from typing import Optional
import re
from config import VALID_MATERIAL, NORMALIZE_KONDISI, KONDISI_ORDER


@dataclass
class ParsedOutput:
    """Representasi hasil parsing satu output model."""

    raw_text: str = ""

    atap_material: Optional[str] = None
    atap_kondisi: Optional[str] = None
    dinding_material: Optional[str] = None
    dinding_kondisi: Optional[str] = None
    lantai_material: Optional[str] = None
    lantai_kondisi: Optional[str] = None

    konflik_dinding: Optional[bool] = None
    penjelasan: Optional[str] = None

    is_valid: bool = False
    parse_errors: list[str] = field(default_factory=list)


# SECTION 2 — PARSER


def parse_model_output(text: str) -> ParsedOutput:
    """
    Mem-parse teks output model menjadi ParsedOutput terstruktur.

    Template yang diharapkan:
        Atap:
        - Material: <nilai>
        - Kondisi: <nilai>
        Dinding:
        - Material: <nilai>
        - Kondisi: <nilai>
        Lantai:
        - Material: <nilai>
        - Kondisi: <nilai>
        Konflik:
        - Dinding: true/false
        Penjelasan:
        <teks bebas>
    """
    result = ParsedOutput(raw_text=text)

    if not text or not isinstance(text, str):
        result.parse_errors.append("Output kosong atau bukan string.")
        return result

    text_clean = text.strip()

    # ── Helper regex extractor ──────────────────────────────────────
    def extract(pattern: str, src: str, flags: int = re.IGNORECASE):
        m = re.search(pattern, src, flags)
        if not m:
            return None
        return m.group(1).split("\n")[0].strip().lower()

    # ── Normalisasi tanda hubung & spasi ────────────────────────────
    text_norm = re.sub(r"[ \t]+", " ", text_clean)

    # ── Ekstrak setiap komponen ──────────────────────────────────────

    result.atap_material = extract(
        r"Atap\s*:.*?Material\s*:\s*(.+)", text_norm, re.I | re.S
    )
    result.atap_kondisi = extract(
        r"Atap\s*:.*?Kondisi\s*:\s*(.+)", text_norm, re.I | re.S
    )

    result.dinding_material = extract(
        r"Dinding\s*:.*?Material\s*:\s*(.+)", text_norm, re.I | re.S
    )
    result.dinding_kondisi = extract(
        r"Dinding\s*:.*?Kondisi\s*:\s*(.+)", text_norm, re.I | re.S
    )

    result.lantai_material = extract(
        r"Lantai\s*:.*?Material\s*:\s*(.+)", text_norm, re.I | re.S
    )
    result.lantai_kondisi = extract(
        r"Lantai\s*:.*?Kondisi\s*:\s*(.+)", text_norm, re.I | re.S
    )

    # ── Konflik: true/false/ya/tidak ────────────────────────────────
    konflik_match = re.search(
        r"Konflik\s*Dinding\s*:\s*(.+)",
        text_norm,
        re.I
    )

    if konflik_match:
        val = konflik_match.group(1).strip().lower()

        result.konflik_dinding = val in ["ya", "true", "1", "benar"]
    else:
        result.parse_errors.append("Field 'Konflik Dinding' tidak ditemukan.")

    # ── Penjelasan: semua teks setelah "Penjelasan:" ─────────────────
    penj_match = re.search(r"Penjelasan\s*:?\s*([\s\S]+)", text_norm, re.IGNORECASE)
    if penj_match:
        result.penjelasan = penj_match.group(1).strip()
    else:
        result.parse_errors.append("Section 'Penjelasan' tidak ditemukan.")

    # ── Validasi field wajib ─────────────────────────────────────────
    required = {
        "atap_material": result.atap_material,
        "atap_kondisi": result.atap_kondisi,
        "dinding_material": result.dinding_material,
        "dinding_kondisi": result.dinding_kondisi,
        "lantai_material": result.lantai_material,
        "lantai_kondisi": result.lantai_kondisi,
        "konflik_dinding": result.konflik_dinding,
        "penjelasan": result.penjelasan,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        result.parse_errors.extend([f"Field hilang: {m}" for m in missing])

    # ── Validasi material ─────────────────────────────────────────
    for comp in ["atap", "dinding", "lantai"]:
        material = getattr(result, f"{comp}_material")
        valid_list = VALID_MATERIAL[comp]

        if material:
            material_clean = material.replace(" ", "_")
            if material_clean not in valid_list:
                result.parse_errors.append(f"Material {comp} tidak valid: {material}")

    # ── Bersihkan whitespace sisa pada nilai kondisi ─────────────────
    for attr in ("atap_kondisi", "dinding_kondisi", "lantai_kondisi"):
        val = getattr(result, attr)
        if val:
            val_clean = val.strip().lower()
            val_clean = val_clean.replace("-", " ").replace("_", " ")

            normalized = NORMALIZE_KONDISI.get(val_clean, val_clean.replace(" ", "_"))
            setattr(result, attr, normalized)

    # ── Validasi kondisi ─────────────────────────────────────────
    for comp in ["atap", "dinding", "lantai"]:
        kondisi = getattr(result, f"{comp}_kondisi")
        if kondisi and kondisi not in KONDISI_ORDER:
            mapped = NORMALIZE_KONDISI.get(kondisi.replace("_", " "), None)
            if mapped:
                setattr(result, f"{comp}_kondisi", mapped)
            else:
                result.parse_errors.append(f"Kondisi {comp} tidak valid: {kondisi}")

    result.is_valid = len(result.parse_errors) == 0
    return result
