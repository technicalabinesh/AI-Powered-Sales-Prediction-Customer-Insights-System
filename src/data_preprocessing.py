"""
Data Preprocessing Module
Handles loading, cleaning, and feature engineering of the sales dataset.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load the sales dataset from a CSV file."""
    df = pd.read_csv(filepath, parse_dates=["OrderDate"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and handle missing values."""
    df = df.drop_duplicates()
    df = df.dropna(subset=["Sales", "OrderDate", "CustomerID"])
    df["Sales"] = df["Sales"].clip(lower=0)
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date-based and encoded features from the cleaned dataset."""
    df = df.copy()

    # Date features
    df["Year"] = df["OrderDate"].dt.year
    df["Month"] = df["OrderDate"].dt.month
    df["Quarter"] = df["OrderDate"].dt.quarter
    df["DayOfWeek"] = df["OrderDate"].dt.dayofweek
    df["WeekOfYear"] = df["OrderDate"].dt.isocalendar().week.astype(int)

    # Encode categorical columns
    for col in ["Region", "Category"]:
        df[col + "_Code"] = df[col].astype("category").cat.codes

    return df


def get_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales by year and month."""
    monthly = (
        df.groupby(["Year", "Month"])["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "TotalSales"})
    )
    monthly["PeriodLabel"] = monthly.apply(
        lambda r: f"{int(r.Year)}-{int(r.Month):02d}", axis=1
    )
    return monthly


def preprocess(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline. Returns (clean_df, monthly_df)."""
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    monthly = get_monthly_sales(df)
    return df, monthly
