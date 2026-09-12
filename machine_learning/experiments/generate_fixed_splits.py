import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fixed train/test split and save row IDs for reproducible model comparisons.")
    parser.add_argument("--csv", required=True, help="CSV dataset to split.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--output-dir", default="machine_learning/splits", help="Directory where split CSV files will be written.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction assigned to the test set.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.csv).expanduser().resolve()
    df = pd.read_csv(dataset_path)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {dataset_path}")

    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.target],
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"row_id": train_idx}).to_csv(output_dir / "train_ids.csv", index=False)
    pd.DataFrame({"row_id": test_idx}).to_csv(output_dir / "test_ids.csv", index=False)

    print(f"Saved splits for {len(df)} rows to {output_dir}")
    print(f"Train rows: {len(train_idx)}")
    print(f"Test rows: {len(test_idx)}")


if __name__ == "__main__":
    main()
