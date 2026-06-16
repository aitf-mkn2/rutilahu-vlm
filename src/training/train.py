from __future__ import annotations

import argparse
import logging
from pathlib import Path

from configs.config import SFTConfig
from src.training.trainer import (
    build_datasets,
    build_model_and_tokenizer,
    build_trainer,
    save_training_artifacts,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Vision Language Model dengan Unsloth"
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/base.yaml",
        help="Path ke configs/base.yaml",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data.yaml",
        help="Path ke configs/data.yaml",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        default="configs/experiment.yaml",
        help="Path ke configs/experiment.yaml",
    )
    parser.add_argument(
        "--qlora_config",
        type=str,
        default="configs/qlora.yaml",
        help="Path ke configs/qlora.yaml",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path checkpoint untuk resume training",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    base_yaml = Path(args.base_config)
    data_yaml = Path(args.data_config)
    experiment_yaml = Path(args.experiment_config)
    qlora_yaml = Path(args.qlora_config)

    cfg = SFTConfig.from_files(
        base_yaml=base_yaml,
        data_yaml=data_yaml,
        experiment_yaml=experiment_yaml,
        qlora_yaml=qlora_yaml,
    )

    logger.info("Experiment name: %s", cfg.experiment.run_name)
    logger.info("HF Repo ID: %s", cfg.base.hf_repo_id)
    logger.info("Config loaded successfully.")
    logger.info("Model name: %s", cfg.base.model_name)
    logger.info("Output dir: %s", cfg.experiment.output_dir)
    logger.info("Image root: %s", cfg.data.image_root)
    logger.info("Learning rate: %s", cfg.experiment.learning_rate)

    Path(cfg.experiment.output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(cfg)
    train_dataset, eval_dataset = build_datasets(cfg)
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if args.resume_from_checkpoint:
        logger.info("Resuming from checkpoint: %s", args.resume_from_checkpoint)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
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

    logger.info("Training selesai dan artifacts sudah disimpan.")


if __name__ == "__main__":
    main()