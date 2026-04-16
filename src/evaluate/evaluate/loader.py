import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


# LOAD JSONL
def load_jsonl(path: Path) -> list[dict]:
    """Muat file JSONL, satu dict per baris."""
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Baris %d di '%s' tidak valid JSON: %s", lineno, path, e)
    return records


# EXTRACT TEXT
def extract_text_from_record(record: dict) -> str | None:
    """
    Ekstrak teks output dari berbagai format record:
    1. {"prediction": "..."}          ← format predictions.jsonl
    2. {"reference": "..."}           ← format references sederhana
    3. {"messages": [...]}            ← format JSONL conversation (HuggingFace)
    4. {"output": "..."}              ← alias umum
    """
    for key in ("prediction", "reference", "output", "response"):
        if key in record and isinstance(record[key], str):
            return record[key]

    # Format conversation HuggingFace
    if "messages" in record:
        for msg in reversed(record["messages"]):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [
                        c.get("text", "") for c in content if c.get("type") == "text"
                    ]
                    return "\n".join(texts).strip() or None
    return None


# EXTRACT ID
def extract_id(record: dict, fallback: int) -> str:
    """Ambil identifier unik dari record."""
    for key in ("id", "sample_id", "image_id", "idx"):
        if key in record:
            return str(record[key])
    return str(fallback)


# ALIGN DATA
def align_predictions_references(
    preds: list[dict], refs: list[dict]
) -> list[tuple[dict, dict]]:
    """
    Pasangkan prediction & reference berdasarkan ID.
    Jika tidak ada field ID, gunakan posisi (index-based).
    """
    pred_has_id = any("id" in r or "sample_id" in r for r in preds)
    ref_has_id = any("id" in r or "sample_id" in r for r in refs)

    # ── ID-based matching ─────────────────────────────
    if pred_has_id and ref_has_id:
        pred_map = {extract_id(r, i): r for i, r in enumerate(preds)}
        ref_map = {extract_id(r, i): r for i, r in enumerate(refs)}
        common = sorted(set(pred_map) & set(ref_map))
        if len(common) < len(ref_map):
            logger.warning(
                "%d reference tidak memiliki pasangan prediction dan akan dilewati.",
                len(ref_map) - len(common),
            )
        return [(pred_map[k], ref_map[k]) for k in common]

    # ── Fallback: index-based ─────────────────────────
    n = min(len(preds), len(refs))

    if n < max(len(preds), len(refs)):
        logger.warning(
            "Jumlah predictions (%d) dan references (%d) berbeda. "
            "Menggunakan %d pasangan pertama.",
            len(preds),
            len(refs),
            n,
        )

    return [(preds[i], refs[i], str(i)) for i in range(n)]
