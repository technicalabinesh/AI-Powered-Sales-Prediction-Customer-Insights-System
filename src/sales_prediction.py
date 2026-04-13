"""
Sales Prediction Module
Trains a Random Forest Regression model to predict monthly sales
and evaluates performance with RMSE and R² metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


MODELS_DIR = "outputs/models"
FIGURES_DIR = "outputs/figures"


def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select feature columns and target for model training."""
    feature_cols = [
        "Month", "Quarter", "DayOfWeek", "WeekOfYear",
        "Region_Code", "Category_Code",
    ]
    X = df[feature_cols]
    y = df["Sales"]
    return X, y


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """
    Train Linear Regression and Random Forest models.
    Returns a dict with model objects and performance metrics.
    """
    _ensure_dirs()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)
    results["LinearRegression"] = {
        "model": lr, "scaler": scaler,
        "rmse": lr_rmse, "r2": lr_r2,
        "y_test": y_test, "preds": lr_preds,
    }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)
    results["RandomForest"] = {
        "model": rf, "scaler": None,
        "rmse": rf_rmse, "r2": rf_r2,
        "y_test": y_test, "preds": rf_preds,
    }

    print("\n[Sales Prediction] Model Performance:")
    for name, res in results.items():
        print(f"  {name:20s}  RMSE={res['rmse']:.2f}  R²={res['r2']:.4f}")

    # Save best model (lowest RMSE)
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best = results[best_name]
    joblib.dump({"model": best["model"], "scaler": best["scaler"]},
                os.path.join(MODELS_DIR, "best_sales_model.pkl"))
    print(f"[Sales Prediction] Best model: {best_name} — saved to {MODELS_DIR}/best_sales_model.pkl")

    _plot_predictions(results, X_train, y_train, rf)
    return results


def _plot_predictions(results: dict, X_train: pd.DataFrame,
                      y_train: pd.Series, rf: RandomForestRegressor):
    """Plot actual vs predicted and feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, res) in zip(axes, results.items()):
        ax.scatter(res["y_test"], res["preds"], alpha=0.6, edgecolors="k", linewidths=0.5)
        lims = [min(res["y_test"].min(), res["preds"].min()),
                max(res["y_test"].max(), res["preds"].max())]
        ax.plot(lims, lims, "r--", linewidth=1.5)
        ax.set_title(f"{name}\nRMSE={res['rmse']:.0f}  R²={res['r2']:.3f}")
        ax.set_xlabel("Actual Sales ($)")
        ax.set_ylabel("Predicted Sales ($)")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)

    # Feature importance (Random Forest)
    importances = pd.Series(rf.feature_importances_,
                            index=X_train.columns).sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    importances.plot(kind="bar", ax=ax2, color="steelblue")
    ax2.set_title("Feature Importance (Random Forest)")
    ax2.set_ylabel("Importance")
    plt.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig2)
    print("[Sales Prediction] Saved actual_vs_predicted.png and feature_importance.png")
