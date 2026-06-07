from __future__ import annotations

import argparse

from src.data.dataset_utils import (
    generate_credit_risk_dataframe,
    save_credit_risk_dataframe,
    summarize_dataframe,
)
from src.utils.common import DATA_OUTPUT_DIR, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic credit-risk dataset for the lab.")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of rows to generate.")
    parser.add_argument("--features", type=int, default=50, help="Number of numeric features to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_OUTPUT_DIR / "credit_risk_synthetic.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directory(DATA_OUTPUT_DIR)

    dataframe = generate_credit_risk_dataframe(
        num_rows=args.rows,
        total_features=args.features,
        seed=args.seed,
    )
    output_path = save_credit_risk_dataframe(dataframe, args.output)
    summary = summarize_dataframe(dataframe)

    print("Synthetic credit-risk dataset generated successfully.")
    print(f"Output path: {output_path}")
    print(f"Rows: {summary['rows']}")
    print(f"Features: {summary['feature_count']}")
    print(f"Default rate: {summary['default_rate']:.4f}")


if __name__ == "__main__":
    main()
