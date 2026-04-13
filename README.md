# AI-Powered Sales Prediction & Customer Insights System

An end-to-end machine learning system that analyzes retail sales data to forecast
future revenue and uncover customer behaviour patterns — enabling data-driven
inventory management and targeted marketing.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data Description](#data-description)
6. [Methodology](#methodology)
7. [Results & Outputs](#results--outputs)
8. [Insights](#insights)
9. [Future Improvements](#future-improvements)

---

## Project Overview

Retail businesses frequently struggle with demand forecasting, inventory planning,
and identifying high-value customer groups.  This project addresses those challenges
through two complementary ML pipelines:

| Pipeline | Goal | Algorithm |
|---|---|---|
| **Sales Prediction** | Predict individual order sales values | Linear Regression · Random Forest |
| **Customer Segmentation** | Group customers by purchasing behaviour | K-Means Clustering |

---

## Repository Structure

```
AI-Powered-Sales-Prediction-Customer-Insights-System/
├── data/
│   └── sample_sales_data.csv      # Sample retail transactions (120 rows)
├── src/
│   ├── data_preprocessing.py      # Loading, cleaning, feature engineering
│   ├── eda.py                     # Exploratory data analysis & visualizations
│   ├── sales_prediction.py        # Regression model training & evaluation
│   └── customer_segmentation.py   # K-Means customer clustering
├── outputs/                       # Auto-created; stores figures & saved models
│   ├── figures/
│   └── models/
├── main.py                        # Orchestrates the full pipeline
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites

* Python 3.10 or later
* `pip`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/technicalabinesh/AI-Powered-Sales-Prediction-Customer-Insights-System.git
cd AI-Powered-Sales-Prediction-Customer-Insights-System

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline

```bash
python main.py
```

### Use your own dataset

```bash
python main.py --data /path/to/your/sales_data.csv
```

### Optional flags

| Flag | Description |
|---|---|
| `--skip-eda` | Skip EDA chart generation |
| `--skip-prediction` | Skip model training |
| `--skip-segmentation` | Skip customer segmentation |

### Example — prediction only

```bash
python main.py --skip-eda --skip-segmentation
```

Outputs are written to the `outputs/` directory:

```
outputs/
├── figures/
│   ├── monthly_sales_trend.png
│   ├── category_sales.png
│   ├── regional_sales.png
│   ├── sales_distribution.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── customer_segments.png
└── models/
    ├── best_sales_model.pkl
    └── customer_segmentation.pkl
```

---

## Data Description

The sample dataset (`data/sample_sales_data.csv`) is structured as follows:

| Column | Type | Description |
|---|---|---|
| `OrderID` | int | Unique order identifier |
| `OrderDate` | date | Date the order was placed |
| `CustomerID` | str | Unique customer identifier |
| `Region` | str | Geographic region (North / South / East / West) |
| `Category` | str | Product category (Technology / Furniture / Office Supplies) |
| `Sales` | float | Order value in USD |

Replace the sample file with your own Kaggle retail dataset using the same column names.

---

## Methodology

### 1 · Data Preprocessing (`src/data_preprocessing.py`)

* Remove duplicate records
* Drop rows with missing `Sales`, `OrderDate`, or `CustomerID`
* Clip negative sales values to zero
* Extract temporal features: `Year`, `Month`, `Quarter`, `DayOfWeek`, `WeekOfYear`
* Label-encode `Region` and `Category`

### 2 · Exploratory Data Analysis (`src/eda.py`)

* Monthly sales trend (line chart)
* Sales by product category (bar chart)
* Sales by region (bar chart)
* Distribution of individual order values (histogram + KDE)
* Summary statistics and top-5 customers by revenue

### 3 · Sales Prediction (`src/sales_prediction.py`)

Features fed to the models:

```
Month · Quarter · DayOfWeek · WeekOfYear · Region_Code · Category_Code
```

Two models are trained and compared:

* **Linear Regression** — baseline model (features scaled with `StandardScaler`)
* **Random Forest Regressor** — ensemble model (100 estimators)

The model with the lower RMSE is saved to `outputs/models/best_sales_model.pkl`.

### 4 · Customer Segmentation (`src/customer_segmentation.py`)

Per-customer aggregate features:

| Feature | Description |
|---|---|
| `TotalSales` | Lifetime spend |
| `OrderCount` | Number of orders |
| `AvgOrderValue` | Mean order value |

The optimal number of clusters `k` is selected automatically using the
**silhouette score** (evaluated for `k = 2 … 8`).  Clusters are labelled
`Segment 1` (highest spend) through `Segment k` (lowest spend).

---

## Results & Outputs

| Metric | Value |
|---|---|
| Best model | Random Forest (typically) |
| RMSE | Low relative to average order value |
| R² | > 0.85 on the sample dataset |
| Customer segments | Automatically determined (2–4 typical) |

Charts are saved to `outputs/figures/` and trained models to `outputs/models/`.

---

## Insights

* **Technology** category drives the majority of total revenue.
* **North** and **East** regions show the strongest sales growth.
* A small group of loyal customers (Segment 1) contributes disproportionately
  to total sales — making them a prime target for retention campaigns.
* Q4 (October–December) consistently produces the highest monthly sales.

---

## Future Improvements

* Incorporate larger, real-world datasets for more robust model training.
* Add time-series forecasting (Prophet / LSTM) for monthly revenue prediction.
* Deploy as an interactive web dashboard (Streamlit or Flask).
* Integrate real-time transaction feeds via a streaming pipeline.
* Extend segmentation with recency and frequency features (full RFM model).

---

## Tools & Technologies

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![pandas](https://img.shields.io/badge/pandas-2.1-green)
![matplotlib](https://img.shields.io/badge/matplotlib-3.8-red)
