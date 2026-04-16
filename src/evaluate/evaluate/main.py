import argparse
import sys
from pathlib import Path
import logging

from pipeline import run_evaluation

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VLM output — AITF Tim MKN-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py \\
      --predictions outputs/predictions/predictions.jsonl \\
      --references  data/test.jsonl \\
      --output_dir  outputs/metrics \\
      --split       test
        """,
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path ke file predictions.jsonl (output model)",
    )
    parser.add_argument(
        "--references",
        required=True,
        type=Path,
        help="Path ke file references/test.jsonl (ground truth)",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("outputs/metrics"),
        type=Path,
        help="Direktori untuk menyimpan metrics dan report",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Nama split yang dievaluasi",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        logger.error("File predictions tidak ditemukan: %s", args.predictions)
        sys.exit(1)
    if not args.references.exists():
        logger.error("File references tidak ditemukan: %s", args.references)
        sys.exit(1)

    run_evaluation(
        predictions_path=args.predictions,
        references_path=args.references,
        output_dir=Path("outputs"),
        split=args.split,
    )


if __name__ == "__main__":
    main()
