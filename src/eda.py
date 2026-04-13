"""
Exploratory Data Analysis (EDA) Module
Generates sales trend, category, and regional analysis charts.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


FIGURES_DIR = "outputs/figures"


def _ensure_output_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_monthly_sales_trend(monthly_df: pd.DataFrame):
    """Line chart of total monthly sales."""
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_df["PeriodLabel"], monthly_df["TotalSales"], marker="o", linewidth=2)
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales ($)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "monthly_sales_trend.png"), dpi=150)
    plt.close(fig)
    print(f"[EDA] Saved monthly_sales_trend.png")


def plot_category_sales(df: pd.DataFrame):
    """Bar chart of total sales per product category."""
    _ensure_output_dir()
    cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=cat_sales.index, y=cat_sales.values, hue=cat_sales.index,
                palette="viridis", legend=False, ax=ax)
    ax.set_title("Sales by Product Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Sales ($)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "category_sales.png"), dpi=150)
    plt.close(fig)
    print(f"[EDA] Saved category_sales.png")


def plot_regional_sales(df: pd.DataFrame):
    """Bar chart of total sales per region."""
    _ensure_output_dir()
    region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=region_sales.index, y=region_sales.values, hue=region_sales.index,
                palette="rocket", legend=False, ax=ax)
    ax.set_title("Sales by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Sales ($)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "regional_sales.png"), dpi=150)
    plt.close(fig)
    print(f"[EDA] Saved regional_sales.png")


def plot_sales_distribution(df: pd.DataFrame):
    """Histogram of individual order sales values."""
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Sales"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Order Sales")
    ax.set_xlabel("Sales ($)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "sales_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"[EDA] Saved sales_distribution.png")


def run_eda(df: pd.DataFrame, monthly_df: pd.DataFrame):
    """Run all EDA plots."""
    plot_monthly_sales_trend(monthly_df)
    plot_category_sales(df)
    plot_regional_sales(df)
    plot_sales_distribution(df)

    print("\n[EDA] Summary Statistics:")
    print(df[["Sales"]].describe().round(2))

    print("\n[EDA] Top 5 Customers by Total Sales:")
    top_customers = df.groupby("CustomerID")["Sales"].sum().nlargest(5)
    print(top_customers)
