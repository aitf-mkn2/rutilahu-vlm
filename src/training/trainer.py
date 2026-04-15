# rutilahu-vlm/src/training/trainer.py

import json
import logging
import random
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from transformers import EarlyStoppingCallback

from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from src.data.dataset import VLMdataset

logger = logging.getLogger(__name__)


# =========================
# Config helpers
# =========================
def _load_yaml(path: str) -> Dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config tidak ditemukan: {path}")

    with path_obj.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Root YAML harus dict: {path}")

    return data


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in update.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_and_merge_configs(
    base_config_path: str,
    qlora_config_path: str,
    exp_config_path: str,
) -> Dict[str, Any]:
    cfg = {}
    cfg = _deep_merge(cfg, _load_yaml(base_config_path))
    cfg = _deep_merge(cfg, _load_yaml(qlora_config_path))
    cfg = _deep_merge(cfg, _load_yaml(exp_config_path))
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_yaml(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def _normalize_sft_keys(raw_sft_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalisasi key agar kompatibel lintas versi TRL/HF.
    Contoh:
    - evaluation_strategy -> eval_strategy (jika versi TRL memakai nama ini)
    """
    cfg = deepcopy(raw_sft_cfg)

    valid_fields = {f.name for f in fields(SFTConfig)}

    # Alias umum antar versi
    if "evaluation_strategy" in cfg and "evaluation_strategy" not in valid_fields and "eval_strategy" in valid_fields:
        cfg["eval_strategy"] = cfg.pop("evaluation_strategy")

    # Jika versi lama masih memakai evaluation_strategy, biarkan saja
    # dan buang eval_strategy jika tidak dikenali.
    if "eval_strategy" in cfg and "eval_strategy" not in valid_fields and "evaluation_strategy" in valid_fields:
        cfg["evaluation_strategy"] = cfg.pop("eval_strategy")

    return cfg


def _filter_sft_kwargs(raw_sft_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ambil hanya key yang valid untuk SFTConfig.
    Key lain diabaikan agar config tetap aman terhadap perubahan versi.
    """
    normalized = _normalize_sft_keys(raw_sft_cfg)
    valid_fields = {f.name for f in fields(SFTConfig)}
    return {k: v for k, v in normalized.items() if k in valid_fields}


def _validate_config(cfg: Dict[str, Any]) -> None:
    """
    Validasi config minimal sebelum training dimulai.
    """
    required_top_level = ["model_name", "output_dir"]
    for key in required_top_level:
        if key not in cfg or cfg[key] in (None, ""):
            raise ValueError(f"Config wajib memiliki `{key}` yang tidak kosong.")

    if "sft" in cfg and not isinstance(cfg["sft"], dict):
        raise ValueError("Section `sft` pada YAML harus berupa dictionary.")

    if "lora_r" in cfg and int(cfg["lora_r"]) <= 0:
        raise ValueError("`lora_r` harus > 0.")

    if "lora_alpha" in cfg and int(cfg["lora_alpha"]) <= 0:
        raise ValueError("`lora_alpha` harus > 0.")

    if "lora_dropout" in cfg:
        dropout = float(cfg["lora_dropout"])
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("`lora_dropout` harus di rentang 0.0 sampai 1.0.")

    if "max_length" in cfg and int(cfg["max_length"]) <= 0:
        raise ValueError("`max_length` harus > 0.")

    if "dataset_name" not in cfg and not any(
        cfg.get(k) for k in ["train_data_path", "validation_data_path", "test_data_path"]
    ):
        raise ValueError(
            "Config harus berisi `dataset_name` atau minimal salah satu dari "
            "`train_data_path` / `validation_data_path` / `test_data_path`."
        )


# =========================
# Dataset wrapper
# =========================
class VisionConversationDataset(Dataset):
    """
    Output:
    {
      "messages": [
        {"role": "user", "content": [...]},
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
      ]
    }
    """

    def __init__(self, raw_dataset: VLMdataset):
        self.raw_dataset = raw_dataset

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]
        images = sample["images"]
        instruction = sample["instruction"].strip()
        output = sample["output"].strip()

        user_content = []
        for image in images:
            user_content.append({"type": "image", "image": image})

        user_content.append({"type": "text", "text": instruction})

        return {
            "messages": [
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": output}],
                },
            ]
        }


# =========================
# Main trainer
# =========================
class VLMExperimentTrainer:
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise ValueError("Config harus berupa dictionary.")

        self.cfg = config
        _validate_config(self.cfg)

        self.seed = int(self.cfg.get("seed", 42))
        set_seed(self.seed)

        self.model_name = self.cfg["model_name"]
        self.output_dir = Path(self.cfg.get("output_dir", "outputs/exp_01"))
        self.checkpoints_dir = ensure_dir(self.output_dir / "checkpoints")
        self.logs_dir = ensure_dir(self.output_dir / "logs")
        self.predictions_dir = ensure_dir(self.output_dir / "predictions")

        self.max_length = int(self.cfg.get("max_length", 2048))
        self.resume_from_checkpoint = self.cfg.get("resume_from_checkpoint")

        self.train_split = self.cfg.get("train_split", "train")
        self.val_split = self.cfg.get("val_split", "validation")
        self.test_split = self.cfg.get("test_split", "test")

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.trainer = None
        self.test_dataset = None

        save_yaml(self.cfg, self.output_dir / "config_snapshot.yaml")

    def _resolve_dataset(self, split_name: str) -> VLMdataset:
        base_path = self.cfg.get("base_path", "")

        split_key = f"{split_name}_data_path"
        if self.cfg.get(split_key):
            return VLMdataset(
                data_path=self.cfg[split_key],
                split="train",
                base_path=base_path,
            )

        if self.cfg.get("dataset_name"):
            return VLMdataset(
                dataset_name=self.cfg["dataset_name"],
                split=split_name,
                base_path=base_path,
            )

        raise ValueError(
            "Config harus berisi `dataset_name` atau salah satu file path per split "
            "(train_data_path / validation_data_path / test_data_path)."
        )

    def _build_datasets(self):
        train_raw = self._resolve_dataset(self.train_split)
        val_raw = self._resolve_dataset(self.val_split)

        train_ds = VisionConversationDataset(train_raw)
        val_ds = VisionConversationDataset(val_raw)

        test_ds = None
        if self.cfg.get("use_test_split", True):
            try:
                test_raw = self._resolve_dataset(self.test_split)
                test_ds = VisionConversationDataset(test_raw)
            except Exception as e:
                logger.warning(f"Test split tidak dipakai: {e}")

        return train_ds, val_ds, test_ds

    def _build_model(self):
        load_in_4bit = bool(self.cfg.get("load_in_4bit", True))
        use_gradient_checkpointing = (
            "unsloth" if bool(self.cfg.get("gradient_checkpointing", True)) else False
        )

        loaded = FastVisionModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_length,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing=use_gradient_checkpointing,
            fast_inference=False,
        )

        if isinstance(loaded, tuple) and len(loaded) == 2:
            model, processor = loaded
        else:
            model = loaded
            processor = getattr(loaded, "processor", None)
            if processor is None:
                raise RuntimeError(
                    "FastVisionModel.from_pretrained tidak mengembalikan processor. "
                    "Sesuaikan dengan versi Unsloth yang Anda pakai."
                )

        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        target_modules = self.cfg.get("target_modules", "all-linear")

        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers=bool(self.cfg.get("finetune_vision_layers", False)),
            finetune_language_layers=bool(self.cfg.get("finetune_language_layers", True)),
            finetune_attention_modules=bool(self.cfg.get("finetune_attention_modules", True)),
            finetune_mlp_modules=bool(self.cfg.get("finetune_mlp_modules", True)),
            r=int(self.cfg.get("lora_r", 16)),
            lora_alpha=int(self.cfg.get("lora_alpha", 32)),
            lora_dropout=float(self.cfg.get("lora_dropout", 0.05)),
            bias=self.cfg.get("lora_bias", "none"),
            target_modules=target_modules,
        )

        return self.model

    def _build_sft_config(self) -> SFTConfig:
        """
        Semua parameter training dibaca dari section `sft` pada YAML.
        Jika ada key penting di top-level, dipakai sebagai fallback.
        """
        sft_raw = deepcopy(self.cfg.get("sft", {}))
        if not isinstance(sft_raw, dict):
            raise ValueError("Section `sft` pada YAML harus berupa dictionary.")

        # Fallback dari top-level config
        sft_raw.setdefault("output_dir", str(self.checkpoints_dir))
        sft_raw.setdefault("seed", self.seed)
        sft_raw.setdefault("run_name", self.cfg.get("run_name", self.output_dir.name))
        sft_raw.setdefault("overwrite_output_dir", bool(self.cfg.get("overwrite_output_dir", True)))
        sft_raw.setdefault("report_to", self.cfg.get("report_to", "none"))
        sft_raw.setdefault("bf16", bool(self.cfg.get("bf16", True)))
        sft_raw.setdefault("fp16", bool(self.cfg.get("fp16", False)))
        sft_raw.setdefault("load_best_model_at_end", bool(self.cfg.get("load_best_model_at_end", True)))
        sft_raw.setdefault("metric_for_best_model", self.cfg.get("metric_for_best_model", "eval_loss"))
        sft_raw.setdefault("greater_is_better", bool(self.cfg.get("greater_is_better", False)))

        # Fallback untuk key yang sering ditaruh di top-level
        for key in [
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "warmup_steps",
            "lr_scheduler_type",
            "logging_steps",
            "eval_steps",
            "save_steps",
            "save_total_limit",
            "save_strategy",
            "evaluation_strategy",
            "eval_strategy",
            "load_best_model_at_end",
            "metric_for_best_model",
            "greater_is_better",
            "bf16",
            "fp16",
            "optim",
            "max_grad_norm",
            "dataloader_num_workers",
            "remove_unused_columns",
            "report_to",
            "dataset_num_proc",
            "overwrite_output_dir",
        ]:
            if key in self.cfg and key not in sft_raw:
                sft_raw[key] = self.cfg[key]

        # Validasi tambahan untuk early stopping
        patience = sft_raw.get("early_stopping_patience")
        if patience is not None:
            patience = int(patience)
            if patience < 1:
                raise ValueError("`early_stopping_patience` harus >= 1.")

            # Early stopping butuh evaluasi berkala
            eval_strategy = sft_raw.get("evaluation_strategy", sft_raw.get("eval_strategy", "steps"))
            if eval_strategy == "no":
                raise ValueError(
                    "`early_stopping_patience` diset tetapi evaluation strategy = 'no'. "
                    "Aktifkan `evaluation_strategy`/`eval_strategy`."
                )

        filtered = _filter_sft_kwargs(sft_raw)

        unknown = sorted(set(sft_raw.keys()) - set(filtered.keys()))
        if unknown:
            logger.warning(
                "Key YAML berikut diabaikan karena bukan field SFTConfig pada versi TRL ini: %s",
                unknown,
            )

        return SFTConfig(**filtered)

    def _build_trainer(self):
        train_ds, val_ds, test_ds = self._build_datasets()
        model = self._build_model()
        sft_config = self._build_sft_config()

        collator = UnslothVisionDataCollator(
            model=model,
            processor=self.processor,
            max_seq_length=self.max_length,
            resize="min",
            completion_only_loss=True,
        )

        callbacks = []
        patience = self.cfg.get("sft", {}).get("early_stopping_patience")
        if patience is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=int(patience)
                )
            )

        trainer_kwargs = dict(
            model=model,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            args=sft_config,
            callbacks=callbacks,
        )

        # Fallback untuk perbedaan versi TRL:
        # - sebagian versi pakai `processing_class`
        # - sebagian versi pakai `tokenizer`
        try:
            trainer = SFTTrainer(
                **trainer_kwargs,
                processing_class=self.processor,
            )
        except TypeError:
            trainer = SFTTrainer(
                **trainer_kwargs,
                tokenizer=self.tokenizer,
            )

        self.trainer = trainer
        self.test_dataset = test_ds
        return trainer

    def _save_model_artifacts(self):
        if self.trainer is None:
            raise RuntimeError("Trainer belum dibuat.")

        final_dir = ensure_dir(self.output_dir / "final_model")
        best_dir = ensure_dir(self.output_dir / "best_model")

        self.trainer.model.save_pretrained(str(final_dir))
        self.processor.save_pretrained(str(final_dir))

        self.trainer.model.save_pretrained(str(best_dir))
        self.processor.save_pretrained(str(best_dir))

        save_json(self.trainer.state.log_history, self.logs_dir / "train_log_history.json")
        save_json(
            {"best_model_checkpoint": self.trainer.state.best_model_checkpoint},
            self.logs_dir / "best_checkpoint.json",
        )

    def train(self):
        trainer = self._build_trainer()

        logger.info("Mulai training...")
        train_result = trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)

        train_metrics = train_result.metrics
        train_metrics["train_samples"] = len(trainer.train_dataset)
        save_json(train_metrics, self.logs_dir / "train_metrics.json")

        logger.info("Evaluasi validation set...")
        eval_metrics = trainer.evaluate()
        save_json(eval_metrics, self.logs_dir / "eval_metrics.json")

        self._save_model_artifacts()

        if self.test_dataset is not None:
            logger.info("Evaluasi test set...")
            test_metrics = trainer.evaluate(eval_dataset=self.test_dataset)
            save_json(test_metrics, self.logs_dir / "test_metrics.json")

        logger.info(f"Selesai. Output tersimpan di {self.output_dir}")
        return {
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
        }