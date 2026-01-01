# Dangote Cement Stock Prediction

## Project overview
End-to-end Python pipeline for forecasting Dangote Cement stock prices and analyzing daily returns. The project includes data ingestion, cleaning, feature engineering, model training/comparison, evaluation, and figure generation.

**Key features:**
- Time-series forecasting (next-day price) and daily returns modeling
- Feature engineering: moving averages (MA_7, MA_30), lag features, daily range, volume categories
- Model comparison: Linear Regression, Ridge, Random Forest, Gradient Boosting, SVR
- Saves trained models (`models/*.joblib`) and figures (`figures/*.pdf`) for reproducibility

## Dataset
- Source: CSV files included in `data/raw/` (e.g., `Dangote Cement Stock Price History.csv`)
- Records: ~1,200 rows (original source file included in repo)
- Core columns: `Date`, `Price`, `Open`, `High`, `Low`, `Vol.`, `Change %`
- Derived features: `MA_7`, `MA_30`, `Lag_n`, `Daily_Range`, `DANGCEM_Daily_Returns`, `Volume_Category`

## Research questions
- **RQ1:** How well can we forecast next-day closing price using lag and technical features?
- **RQ2:** Can we accurately predict daily returns and analyze the predictive power of price-based features?
- **RQ3:** Are there seasonal/monthly patterns in returns and price movement?
- **RQ4:** How does trading volume correlate with daily price range and volatility?

## Installation
1. Create & activate a virtual environment (recommended):
   - Windows PowerShell:
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
     - `pip install --upgrade pip`
     - `pip install -r requirements.txt`

2. (Optional) Pin installed versions:
   - `.venv\Scripts\python -m pip freeze > requirements.lock`

Required packages include: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib` (see `requirements.txt`).

## How to run
- From project root (recommended):
  - `python run.py`  # runs the full pipeline in order
- Or run directly:
  - `python src/main.py`

Outputs:
- Cleaned data: `data/processed/dangcem_cleanedx2.csv`
- Trained models: `models/*.joblib` (e.g., `models/Linear_Regression.joblib`)
- Figures: `figures/*.pdf` (time series, correlation, model comparisons, RQ plots)
- Final terminal message confirms completion and locations of outputs.


## Folder structure
```
Dangote_Stock_Prediction/
├── data/
│   ├── raw/            # raw CSVs (tracked in repo)
│   └── processed/      # cleaned and processed output
├── figures/            # generated figures (PDF)
├── models/             # saved models (.joblib)
├── src/                # source code: ingest, clean, train, visualization, features, evaluation
├── tests/              # (removed) tests were removed by request
├── run.py              # convenience wrapper to run the pipeline
├── requirements.txt
└── README.md
```

## Pipeline stages
1. **Ingest** — `ingest.load_raw()` reads CSVs from `data/raw/`.
2. **Clean** — `clean.clean_stock_data()` normalizes types, computes MA, etc., and saves processed CSV.
3. **Feature engineering** — lag features, moving averages, daily range, volume category.
4. **Train** — `train.train_models_dict()` fits multiple models and saves them to `models/`.
5. **Evaluate** — compute MSE/R^2 and compare model performance.
6. **Visualize** — save figures to `figures/` for RQ plots and EDA.

## Model configuration
- Models included: Linear Regression, Ridge (alpha=1.0), Random Forest (n_estimators=100), Gradient Boosting (n_estimators=100), SVR (rbf)
- Train/test: sequential split (time-series split) 80% train / 20% test

## Reproducibility
To regenerate all outputs:
```bash
rm -rf figures/* models/* data/processed/dangcem_cleanedx2.csv
python run.py
```

## Notes & troubleshooting
- The pipeline prints a completion message like:
  `Pipeline complete. 8 figure(s) generated and saved to 'figures/'. Models saved to 'models/'. Processed data saved to 'data/processed/dangcem_cleanedx2.csv'.`
- If models or figures are missing, ensure `data/raw/` contains the expected CSV and rerun `python run.py`.