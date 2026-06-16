from __future__ import annotations

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from trl import SFTConfig as TRLSFTConfig
from trl import SFTTrainer

import torch
import wandb
import gc
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from configs.config import SFTConfig as AppConfig
from src.data.dataset import MultimodalChatDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _resolve_output_dir(cfg: AppConfig) -> Path:
    """
    Prioritas:
    1) experiment.output_dir jika ada
    2) base.output_dir
    """
    exp_output_dir = getattr(cfg.experiment, "output_dir", None)
    base_output_dir = getattr(cfg.base, "output_dir", None)

    output_dir = exp_output_dir or base_output_dir
    if not output_dir:
        raise ValueError("Output directory tidak ditemukan di config.")

    return Path(output_dir)


def _resolve_report_to(cfg: AppConfig):
    report_to = getattr(cfg.experiment, "report_to", None)
    if report_to is None:
        report_to = getattr(cfg.base, "report_to", None)

    if not report_to:
        return "none"
    return report_to


def _resolve_max_length(cfg: AppConfig) -> int:
    max_length = getattr(cfg.experiment, "max_length", None)
    if max_length is None:
        max_length = getattr(cfg.base, "max_length", 4096)
    return int(max_length)


def _copy_yaml_configs_to_output(
    output_dir: Path,
    base_yaml: Path,
    data_yaml: Path,
    experiment_yaml: Path,
    qlora_yaml: Path,
) -> None:
    """
    Simpan snapshot YAML agar ikut terbawa ke output_dir / Hub.
    """
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for src in [base_yaml, data_yaml, experiment_yaml, qlora_yaml]:
        if not src.exists():
            raise FileNotFoundError(f"Config file tidak ditemukan: {src}")
        shutil.copy2(src, config_dir / src.name)


def _safe_cuda_report() -> str:
    if not torch.cuda.is_available():
        return "CPU only"

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return f"allocated={allocated:.2f} GB | reserved={reserved:.2f} GB"


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_model_and_tokenizer(cfg: AppConfig) -> Tuple[torch.nn.Module, Any]:
    """
    Load Qwen-VL / vision model via Unsloth FastVisionModel.
    """
    max_length = _resolve_max_length(cfg)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=cfg.base.model_name,
        max_seq_length=max_length,
        load_in_4bit=cfg.qlora.load_in_4bit,
        use_gradient_checkpointing="unsloth" if cfg.qlora.gradient_checkpointing else False,
    )

    target_modules = cfg.qlora.target_modules or "all-linear"

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.qlora.finetune_vision_layers,
        finetune_language_layers=cfg.qlora.finetune_language_layers,
        finetune_attention_modules=cfg.qlora.finetune_attention_modules,
        finetune_mlp_modules=cfg.qlora.finetune_mlp_modules,
        r=cfg.qlora.lora_r,
        lora_alpha=cfg.qlora.lora_alpha,
        lora_dropout=cfg.qlora.lora_dropout,
        bias=cfg.qlora.lora_bias,
        target_modules=target_modules,
        random_state=cfg.base.seed,
        max_seq_length=max_length,
    )

    model.config.use_cache = False
    return model, tokenizer


def build_datasets(cfg: AppConfig) -> Tuple[MultimodalChatDataset, Optional[MultimodalChatDataset]]:
    """
    Unsloth vision trainer bisa langsung memakai dataset berisi messages + image object.
    Jadi dataset training dipakai langsung di SFTTrainer.
    """

    dataset_paths = {
        split: str(Path(cfg.data.dataset_root) / filename)
        for split, filename in cfg.data.splits.items()
    }

    train_dataset = MultimodalChatDataset(
        data_path=dataset_paths,
        split="train",
        image_root=cfg.data.image_root,
        cache_images=cfg.data.cache_images,
        cache_size=cfg.data.cache_size,
        verify_images=True,
        strict_validation=True,
        debug_mode=False,
    )

    eval_dataset = None
    if "validation" in cfg.data.splits:
        eval_dataset = MultimodalChatDataset(
            data_path=dataset_paths,
            split="validation",
            image_root=cfg.data.image_root,
            cache_images=cfg.data.cache_images,
            cache_size=cfg.data.cache_size,
            verify_images=True,
            strict_validation=True,
            debug_mode=False,
        )

    return train_dataset, eval_dataset


def build_collator(model, processor, cfg):
    """
    Unsloth vision fine-tuning path.
    """

    return UnslothVisionDataCollator(
        model=model,
        processor=processor,
        max_seq_length=cfg.experiment.max_length,
        resize="min",
        completion_only_loss=True,
    )


def build_trainer(
    cfg: AppConfig,
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataset: MultimodalChatDataset,
    eval_dataset: Optional[MultimodalChatDataset] = None,
) -> SFTTrainer:
    """
    Bangun SFTTrainer untuk vision SFT.
    """
    has_eval = eval_dataset is not None
    eval_strategy = getattr(cfg.experiment, "evaluation_strategy", "no") if has_eval else "no"

    load_best = bool(getattr(cfg.experiment, "load_best_model_at_end", False) and has_eval)
    if not has_eval and getattr(cfg.experiment, "load_best_model_at_end", False):
        logger.warning("load_best_model_at_end dimatikan karena eval_dataset tidak tersedia.")

    output_dir = _resolve_output_dir(cfg)

    trainer_args = TRLSFTConfig(
        output_dir=str(output_dir),
        max_length=_resolve_max_length(cfg),
        num_train_epochs=float(cfg.experiment.num_train_epochs),
        per_device_train_batch_size=int(cfg.experiment.per_device_train_batch_size),
        per_device_eval_batch_size=int(cfg.experiment.per_device_eval_batch_size),
        gradient_accumulation_steps=int(cfg.experiment.gradient_accumulation_steps),
        learning_rate=float(cfg.experiment.learning_rate),
        warmup_ratio=float(cfg.experiment.warmup_ratio),
        lr_scheduler_type=str(cfg.experiment.lr_scheduler_type),
        logging_steps=int(cfg.experiment.logging_steps),
        eval_strategy=eval_strategy,
        eval_steps=int(cfg.experiment.eval_steps) if has_eval else None,
        save_strategy=str(cfg.experiment.save_strategy),
        save_steps=int(cfg.experiment.save_steps),
        save_total_limit=int(cfg.experiment.save_total_limit),
        load_best_model_at_end=load_best,
        metric_for_best_model=str(cfg.experiment.metric_for_best_model) if load_best else None,
        greater_is_better=bool(cfg.experiment.greater_is_better) if load_best else None,
        bf16=bool(cfg.experiment.bf16),
        fp16=bool(cfg.experiment.fp16),
        optim=str(cfg.experiment.optim),
        max_grad_norm=float(cfg.experiment.max_grad_norm),
        dataloader_num_workers=int(cfg.experiment.dataloader_num_workers),
        remove_unused_columns=False,  
        report_to=_resolve_report_to(cfg),
        run_name=getattr(cfg.experiment, "run_name", None),
        seed=int(cfg.base.seed),
        assistant_only_loss=False,
        gradient_checkpointing=bool(cfg.qlora.gradient_checkpointing),
        push_to_hub=bool(cfg.base.hf_repo_id),
        hub_model_id=cfg.base.hf_repo_id if cfg.base.hf_repo_id else None,
        hub_private_repo=bool(cfg.base.hf_private) if cfg.base.hf_repo_id else None,
        hub_strategy="end" if cfg.base.hf_repo_id else "every_save",
        max_steps=-1
    )

    data_collator = build_collator(
        model=model,
        processor=tokenizer,
        cfg=cfg,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=trainer_args,
    )

    return trainer


def inspect_first_batch(trainer: SFTTrainer) -> None:
    """
    Debug helper untuk memastikan batch multimodal keluar dengan shape yang benar.
    """
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))

    logger.info("First batch keys: %s", list(batch.keys()))
    for key, value in batch.items():
        if torch.is_tensor(value):
            logger.info("%s shape: %s dtype=%s", key, tuple(value.shape), value.dtype)
        else:
            logger.info("%s type: %s", key, type(value))


def save_training_artifacts(
    cfg: AppConfig,
    trainer: SFTTrainer,
    tokenizer: Any,
    base_yaml: Path,
    data_yaml: Path,
    experiment_yaml: Path,
    qlora_yaml: Path,
) -> None:
    """
    Simpan model, tokenizer/processor, dan snapshot YAML.
    Jika push_to_hub=True dan hub_model_id terisi, save_model() akan memicu push ke Hub.
    """
    output_dir = _resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_yaml_configs_to_output(output_dir, base_yaml, data_yaml, experiment_yaml, qlora_yaml)

    # Simpan tokenizer / processor agar inference bisa direload dengan benar.
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(output_dir))

    # save_model() juga akan memicu push ke Hub jika push_to_hub aktif.
    trainer.save_model(str(output_dir))


def run_training(
    cfg: AppConfig,
    base_yaml: Path,
    data_yaml: Path,
    experiment_yaml: Path,
    qlora_yaml: Path,
) -> None:
    output_dir = _resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("CUDA status: %s", _safe_cuda_report())
    logger.info("Output dir: %s", output_dir)
    logger.info("Model: %s", cfg.base.model_name)

    wandb_project = "rutilahu-vlm"
    if cfg.base.hf_repo_id:
        wandb_project = cfg.base.hf_repo_id.split("/")[-1]

    wandb.init(
        project=wandb_project,
        name=cfg.experiment.run_name,
        config={
            "model_name": cfg.base.model_name,
            "learning_rate": cfg.experiment.learning_rate,
            "max_length": cfg.experiment.max_length,
            "experiment_name": cfg.experiment_name,
        },
    )

    try:
        model, tokenizer = build_model_and_tokenizer(cfg)
        train_dataset, eval_dataset = build_datasets(cfg)
        trainer = build_trainer(cfg, model, tokenizer, train_dataset, eval_dataset)

        if getattr(cfg.experiment, "debug_first_batch", True):
            inspect_first_batch(trainer)

        trainer.train()

        save_training_artifacts(
            cfg=cfg,
            trainer=trainer,
            tokenizer=tokenizer,
            base_yaml=base_yaml,
            data_yaml=data_yaml,
            experiment_yaml=experiment_yaml,
            qlora_yaml=qlora_yaml,
        )

        logger.info("Training selesai. Model dan tokenizer telah disimpan ke %s", output_dir)
    finally:
        wandb.finish()

    _cleanup()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml", type=str, default="configs/base.yaml")
    parser.add_argument("--data_yaml", type=str, default="configs/data.yaml")
    parser.add_argument("--experiment_yaml", type=str, default="configs/experiment.yaml")
    parser.add_argument("--qlora_yaml", type=str, default="configs/qlora.yaml")
    args = parser.parse_args()

    base_yaml = Path(args.base_yaml)
    data_yaml = Path(args.data_yaml)
    experiment_yaml = Path(args.experiment_yaml)
    qlora_yaml = Path(args.qlora_yaml)

    cfg = AppConfig.from_files(
        base_yaml=base_yaml,
        data_yaml=data_yaml,
        experiment_yaml=experiment_yaml,
        qlora_yaml=qlora_yaml,
    )

    run_training(
        cfg=cfg,
        base_yaml=base_yaml,
        data_yaml=data_yaml,
        experiment_yaml=experiment_yaml,
        qlora_yaml=qlora_yaml,
    )


if __name__ == "__main__":
    main()