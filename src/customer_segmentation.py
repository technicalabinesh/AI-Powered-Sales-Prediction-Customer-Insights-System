"""
Customer Segmentation Module
Applies K-Means clustering to identify distinct customer segments
based on purchasing behavior (RFM-inspired features).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


MODELS_DIR = "outputs/models"
FIGURES_DIR = "outputs/figures"


def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-customer aggregated features:
      - TotalSales   : sum of all order values
      - OrderCount   : number of orders
      - AvgOrderValue: mean order value
    """
    customer_df = (
        df.groupby("CustomerID")
        .agg(
            TotalSales=("Sales", "sum"),
            OrderCount=("OrderID", "count"),
            AvgOrderValue=("Sales", "mean"),
        )
        .reset_index()
    )
    return customer_df


def find_optimal_k(customer_features: pd.DataFrame,
                   scaled_features: np.ndarray,
                   max_k: int = 8) -> int:
    """Use silhouette score to determine the best number of clusters."""
    feature_cols = ["TotalSales", "OrderCount", "AvgOrderValue"]
    n_samples = len(customer_features[feature_cols])
    max_k = min(max_k, n_samples - 1)

    scores = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_features)
        scores[k] = silhouette_score(scaled_features, labels)

    best_k = max(scores, key=scores.get)
    print(f"[Segmentation] Silhouette scores: {scores}")
    print(f"[Segmentation] Optimal k = {best_k}")
    return best_k


def segment_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer features, run K-Means, and return the customer
    DataFrame enriched with a Segment label.
    """
    _ensure_dirs()
    customer_df = build_customer_features(df)
    feature_cols = ["TotalSales", "OrderCount", "AvgOrderValue"]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(customer_df[feature_cols])

    best_k = find_optimal_k(customer_df, scaled)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    customer_df["Cluster"] = kmeans.fit_predict(scaled)

    # Label clusters by average total sales (descending)
    cluster_rank = (
        customer_df.groupby("Cluster")["TotalSales"]
        .mean()
        .rank(ascending=False)
        .astype(int)
    )
    label_map = {c: f"Segment {r}" for c, r in cluster_rank.items()}
    customer_df["Segment"] = customer_df["Cluster"].map(label_map)

    # Save model
    joblib.dump({"kmeans": kmeans, "scaler": scaler},
                os.path.join(MODELS_DIR, "customer_segmentation.pkl"))

    print("\n[Segmentation] Cluster Summary:")
    print(
        customer_df.groupby("Segment")[feature_cols]
        .mean()
        .round(2)
        .to_string()
    )

    _plot_segments(customer_df)
    return customer_df


def _plot_segments(customer_df: pd.DataFrame):
    """Scatter plot of customer segments (TotalSales vs OrderCount)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for segment, group in customer_df.groupby("Segment"):
        ax.scatter(
            group["OrderCount"],
            group["TotalSales"],
            label=segment,
            s=80,
            alpha=0.8,
        )
    ax.set_title("Customer Segmentation (K-Means)")
    ax.set_xlabel("Order Count")
    ax.set_ylabel("Total Sales ($)")
    ax.legend(title="Segment")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "customer_segments.png"), dpi=150)
    plt.close(fig)
    print("[Segmentation] Saved customer_segments.png")
