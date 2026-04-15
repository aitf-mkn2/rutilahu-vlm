import argparse

from trainer import VLMExperimentTrainer, load_and_merge_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL 8B dengan Unsloth")
    parser.add_argument("--base_config", type=str, required=True, help="Path ke configs/base.yaml")
    parser.add_argument("--qlora_config", type=str, required=True, help="Path ke configs/qlora.yaml")
    parser.add_argument("--exp_config", type=str, required=True, help="Path ke configs/experiment.yaml")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path checkpoint untuk resume training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_and_merge_configs(
        base_config_path=args.base_config,
        qlora_config_path=args.qlora_config,
        exp_config_path=args.exp_config,
    )

    if args.resume_from_checkpoint:
        cfg["resume_from_checkpoint"] = args.resume_from_checkpoint

    runner = VLMExperimentTrainer(cfg)
    runner.train()


if __name__ == "__main__":
    main()