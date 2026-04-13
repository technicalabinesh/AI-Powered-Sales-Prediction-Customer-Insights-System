"""
AI-Powered Sales Prediction & Customer Insights System
Main entry point — orchestrates preprocessing, EDA, prediction, and segmentation.
"""

import argparse
import sys
import os

# Allow running from the project root without installing as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import preprocess
from eda import run_eda
from sales_prediction import train_and_evaluate
from customer_segmentation import segment_customers

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_sales_data.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Powered Sales Prediction & Customer Insights System"
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to the sales CSV file (default: data/sample_sales_data.csv)",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip exploratory data analysis charts",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip sales prediction model training",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip customer segmentation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  AI-Powered Sales Prediction & Customer Insights System")
    print("=" * 60)

    # 1. Preprocessing
    print("\n[1/4] Loading and preprocessing data …")
    df, monthly_df = preprocess(args.data)
    print(f"      Records loaded : {len(df)}")
    print(f"      Unique customers: {df['CustomerID'].nunique()}")
    print(f"      Date range      : {df['OrderDate'].min().date()} → {df['OrderDate'].max().date()}")

    # 2. EDA
    if not args.skip_eda:
        print("\n[2/4] Running exploratory data analysis …")
        run_eda(df, monthly_df)
    else:
        print("\n[2/4] EDA skipped.")

    # 3. Sales prediction
    if not args.skip_prediction:
        print("\n[3/4] Training sales prediction models …")
        train_and_evaluate(df)
    else:
        print("\n[3/4] Sales prediction skipped.")

    # 4. Customer segmentation
    if not args.skip_segmentation:
        print("\n[4/4] Running customer segmentation …")
        segment_customers(df)
    else:
        print("\n[4/4] Customer segmentation skipped.")

    print("\n" + "=" * 60)
    print("  Done! Outputs saved to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
