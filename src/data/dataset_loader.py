from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Union

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

DataPathType = Union[str, Path, Dict[str, Union[str, Path]]]


def _as_path(value: Union[str, Path]) -> Path:
    """Convert string / Path to Path object."""
    return value if isinstance(value, Path) else Path(value)


def _validate_jsonl_path(path: Path) -> None:
    """Validasi file JSONL lokal."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset JSONL tidak ditemukan: {path}")

    if not path.is_file():
        raise ValueError(f"Path dataset bukan file: {path}")

    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Dataset harus berformat .jsonl, tetapi mendapat: {path}")


def _resolve_split_path(data_path: DataPathType, split: str) -> Path:
    """
    Resolve path untuk split tertentu.

    Support:
    - string / Path => dianggap file JSONL langsung
    - dict split -> path
    """
    if isinstance(data_path, dict):
        if split not in data_path:
            raise KeyError(
                f"Split `{split}` tidak ditemukan di data_path. "
                f"Key tersedia: {list(data_path.keys())}"
            )
        return _as_path(data_path[split]).expanduser()

    return _as_path(data_path).expanduser()


def load_jsonl(path: Union[str, Path]) -> Dataset:
    """
    Load satu file JSONL lokal menjadi Hugging Face Dataset.
    """
    jsonl_path = _as_path(path).expanduser()
    _validate_jsonl_path(jsonl_path)

    logger.info("Loading JSONL: %s", jsonl_path)

    dataset = load_dataset(
        "json",
        data_files={"train": str(jsonl_path)},
        split="train",
    )

    logger.info("Loaded %d samples from %s", len(dataset), jsonl_path)
    return dataset


def load_dataset_split(
    data_path: DataPathType,
    split: str = "train",
) -> Dataset:
    """
    Load dataset untuk split tertentu.
    """
    resolved_path = _resolve_split_path(data_path, split)
    return load_jsonl(resolved_path)