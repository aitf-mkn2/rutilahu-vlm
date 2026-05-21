from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
import re


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _require(mapping: Dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key `{key}` in section `{section}`")
    return mapping[key]


def _normalize_report_to(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"", "none", "null"}:
            return []
        return [v]

    return list(value)


def _normalize_target_modules(value: Any) -> List[str]:
    """
    Mendukung:
    - list eksplisit
    - string "all-linear" untuk Qwen/Qwen-like setup
    """
    if value is None:
        return []

    if isinstance(value, str):
        if value == "all-linear":
            return [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        return [value]

    return list(value)

    import re


def simplify_model_name(model_name: str) -> str:
    """
    Convert model name menjadi short readable key.

    Contoh:
    Qwen/Qwen3-VL-8B-Instruct
    -> qwen3vl8b
    """

    model_key = model_name.split("/")[-1]
    model_key = model_key.lower()

    model_key = model_key.replace("-instruct", "")
    model_key = model_key.replace("-", "")

    return model_key


def format_learning_rate(lr: float) -> str:
    """
    Convert learning rate ke compact scientific notation.

    Contoh:
    0.00002 -> 2e5
    """

    scientific = f"{lr:.0e}"

    # 2e-05 -> 2e5
    scientific = scientific.replace("e-0", "e")
    scientific = scientific.replace("e-", "e")

    return scientific


def build_experiment_name(cfg) -> str:
    """
    Build automatic experiment name.

    Format:
    mkn2-<base_model>-lr<learning_rate>
    """

    model_key = simplify_model_name(cfg.base.model_name)
    lr_key = format_learning_rate(cfg.experiment.learning_rate)

    return f"mkn2-{model_key}-lr{lr_key}"


@dataclass(frozen=True)
class BaseConfig:
    model_name: str
    output_dir: str
    seed: int
    hf_repo_id: str
    hf_private: bool


@dataclass(frozen=True)
class DataConfig:
    source: str
    splits: Dict[str, str]
    image_roots: Dict[str, str]
    use_test_split: bool
    cache_images: bool
    cache_size: int
    dataset_name: str
    hf_splits: Dict[str, str]

    @property
    def image_root(self) -> str:
        """
        Pilih image root berdasarkan `source`.
        """
        if self.source in self.image_roots:
            return self.image_roots[self.source]

        raise KeyError(
            f"Image root for source `{self.source}` not found in `image_roots`."
        )


@dataclass(frozen=True)
class ExperimentConfig:
    max_length: int
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    lr_scheduler_type: str
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    save_strategy: str
    evaluation_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    bf16: bool
    fp16: bool
    optim: str
    max_grad_norm: float
    dataloader_num_workers: int
    remove_unused_columns: bool
    overwrite_output_dir: bool
    report_to: List[str] = field(default_factory=list)
    debug_first_batch: bool = True
    run_name: Optional[str] = None
    save_safetensors: bool = True


@dataclass(frozen=True)
class QLoRAConfig:
    load_in_4bit: bool
    gradient_checkpointing: bool
    finetune_vision_layers: bool
    finetune_language_layers: bool
    finetune_attention_modules: bool
    finetune_mlp_modules: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    target_modules: List[str]


@dataclass(frozen=True)
class SFTConfig:
    base: BaseConfig
    data: DataConfig
    experiment: ExperimentConfig
    qlora: QLoRAConfig

    @classmethod
    def from_files(
        cls,
        base_yaml: Union[str, Path],
        data_yaml: Union[str, Path],
        experiment_yaml: Union[str, Path],
        qlora_yaml: Union[str, Path],
    ) -> "SFTConfig":
        base_raw = _load_yaml(base_yaml)
        data_raw = _load_yaml(data_yaml)
        exp_raw = _load_yaml(experiment_yaml)
        qlora_raw = _load_yaml(qlora_yaml)

        data_section = _require(data_raw, "data", "data.yaml")
        sft_section = _require(exp_raw, "sft", "experiment.yaml")

        base = BaseConfig(
            model_name=_require(base_raw, "model_name", "base.yaml"),
            output_dir=_require(base_raw, "output_dir", "base.yaml"),
            seed=int(_require(base_raw, "seed", "base.yaml")),
            hf_repo_id=str(_require(base_raw, "hf_repo_id", "base.yaml")),
            hf_private=bool(_require(base_raw, "hf_private", "base.yaml")),
        )

        data = DataConfig(
            source=str(_require(data_section, "source", "data.data")),
            splits=dict(_require(data_section, "splits", "data.data")),
            image_roots=dict(_require(data_section, "image_roots", "data.data")),
            use_test_split=bool(_require(data_section, "use_test_split", "data.data")),
            cache_images=bool(_require(data_section, "cache_images", "data.data")),
            cache_size=int(_require(data_section, "cache_size", "data.data")),
            dataset_name=str(_require(data_section, "dataset_name", "data.data")),
            hf_splits=dict(_require(data_section, "hf_splits", "data.data")),
        )

        experiment = ExperimentConfig(
            max_length=int(_require(sft_section, "max_length", "experiment.sft")),
            num_train_epochs=float(_require(sft_section, "num_train_epochs", "experiment.sft")),
            per_device_train_batch_size=int(
                _require(sft_section, "per_device_train_batch_size", "experiment.sft")
            ),
            per_device_eval_batch_size=int(
                _require(sft_section, "per_device_eval_batch_size", "experiment.sft")
            ),
            gradient_accumulation_steps=int(
                _require(sft_section, "gradient_accumulation_steps", "experiment.sft")
            ),
            learning_rate=float(_require(sft_section, "learning_rate", "experiment.sft")),
            warmup_steps=int(_require(sft_section, "warmup_steps", "experiment.sft")),
            lr_scheduler_type=str(_require(sft_section, "lr_scheduler_type", "experiment.sft")),
            logging_steps=int(_require(sft_section, "logging_steps", "experiment.sft")),
            eval_steps=int(_require(sft_section, "eval_steps", "experiment.sft")),
            save_steps=int(_require(sft_section, "save_steps", "experiment.sft")),
            save_total_limit=int(_require(sft_section, "save_total_limit", "experiment.sft")),
            save_strategy=str(_require(sft_section, "save_strategy", "experiment.sft")),
            evaluation_strategy=str(_require(sft_section, "evaluation_strategy", "experiment.sft")),
            load_best_model_at_end=bool(
                _require(sft_section, "load_best_model_at_end", "experiment.sft")
            ),
            metric_for_best_model=str(
                _require(sft_section, "metric_for_best_model", "experiment.sft")
            ),
            greater_is_better=bool(
                _require(sft_section, "greater_is_better", "experiment.sft")
            ),
            bf16=bool(_require(sft_section, "bf16", "experiment.sft")),
            fp16=bool(_require(sft_section, "fp16", "experiment.sft")),
            optim=str(_require(sft_section, "optim", "experiment.sft")),
            max_grad_norm=float(_require(sft_section, "max_grad_norm", "experiment.sft")),
            dataloader_num_workers=int(
                _require(sft_section, "dataloader_num_workers", "experiment.sft")
            ),
            remove_unused_columns=bool(
                _require(sft_section, "remove_unused_columns", "experiment.sft")
            ),
            overwrite_output_dir=bool(
                _require(sft_section, "overwrite_output_dir", "experiment.sft")
            ),
            report_to=_normalize_report_to(_require(sft_section, "report_to", "experiment.sft")),
            debug_first_batch=bool(
                _require(sft_section, "debug_first_batch", "experiment.sft")
            ),
            run_name=sft_section.get("run_name"),
            save_safetensors=bool(sft_section.get("save_safetensors", True)),
        )

        qlora = QLoRAConfig(
            load_in_4bit=bool(_require(qlora_raw, "load_in_4bit", "qlora.yaml")),
            gradient_checkpointing=bool(
                _require(qlora_raw, "gradient_checkpointing", "qlora.yaml")
            ),
            finetune_vision_layers=bool(
                _require(qlora_raw, "finetune_vision_layers", "qlora.yaml")
            ),
            finetune_language_layers=bool(
                _require(qlora_raw, "finetune_language_layers", "qlora.yaml")
            ),
            finetune_attention_modules=bool(
                _require(qlora_raw, "finetune_attention_modules", "qlora.yaml")
            ),
            finetune_mlp_modules=bool(
                _require(qlora_raw, "finetune_mlp_modules", "qlora.yaml")
            ),
            lora_r=int(_require(qlora_raw, "lora_r", "qlora.yaml")),
            lora_alpha=int(_require(qlora_raw, "lora_alpha", "qlora.yaml")),
            lora_dropout=float(_require(qlora_raw, "lora_dropout", "qlora.yaml")),
            lora_bias=str(_require(qlora_raw, "lora_bias", "qlora.yaml")),
            target_modules=_normalize_target_modules(
                _require(qlora_raw, "target_modules", "qlora.yaml")
            ),
        )

        return cls(base=base, data=data, experiment=experiment, qlora=qlora)

    @property
    def model_name(self) -> str:
        return self.base.model_name

    @property
    def output_dir(self) -> str:
        return self.base.output_dir

    @property
    def seed(self) -> int:
        return self.base.seed

    @property
    def image_root(self) -> str:
        return self.data.image_root