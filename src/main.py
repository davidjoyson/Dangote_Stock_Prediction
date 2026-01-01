from ingest import load_raw
from clean import clean_stock_data, save_processed
from train import prepare_lag_features, train_basic_lr, prepare_next_price_training, train_models_dict, train_daily_returns_model
from visualization import (
    plot_time_series,
    plot_correlation,
    plot_model_comparison,
    plot_rq1,
    plot_rq2,
    plot_rq3,
    plot_rq4,
)

import pandas as pd


def main():
    # Ingest
    raw = load_raw('data/raw/Dangote Cement Stock Price History.csv')
    print(raw.info())
    print(raw.head())

    # Clean
    df_clean = clean_stock_data(raw)
    save_processed(df_clean, 'data/processed/dangcem_cleanedx2.csv')
    print(f"Dataframe saved to data/processed/dangcem_cleanedx2.csv")

    # Visualizations for EDA (save-only, no display)
    figs = []
    figs.append(plot_time_series(df_clean, out_path='figures/TimeSeries_Fig1.pdf'))
    figs.append(plot_correlation(df_clean, out_path='figures/Correlation_Fig1.pdf'))

    # Basic lag model
    df_for_lags = pd.read_csv('data/processed/dangcem_cleanedx2.csv')
    df_for_lags['Date'] = pd.to_datetime(df_for_lags['Date'])
    df_for_lags.set_index('Date', inplace=True)
    X, y = prepare_lag_features(df_for_lags)
    model, X_train, X_test, y_train, y_test, preds = train_basic_lr(X, y)

    # Next-day price models
    df_model = pd.read_csv('data/processed/dangcem_cleanedx2.csv')
    X_train2, X_test2, y_train2, y_test2 = prepare_next_price_training(df_model)
    models, predictions, results_df = train_models_dict(X_train2, y_train2, X_test2, y_test2)

    print("Model Performance Comparison:")
    print(results_df.to_markdown(index=False))

    # Visualizations: model comparison and RQ plots
    figs.append(plot_model_comparison(y_test2, predictions, out_path='figures/Model_comparison_Fig1.pdf'))
    figs.append(plot_rq1(y_test2, predictions['Linear Regression'], out_path='figures/RQ1_Fig1.pdf'))

    # RQ2: daily returns model
    df_regression = pd.read_csv('data/processed/dangcem_cleanedx2.csv')
    df_regression['Date'] = pd.to_datetime(df_regression['Date'])
    df_regression.set_index('Date', inplace=True)
    model_dr, X_train_dr, X_test_dr, y_train_dr, y_test_dr, preds_dr, mse_dr, r2_dr = train_daily_returns_model(df_regression)
    print(f"Multiple Linear Regression Model Performance:\n  Mean Squared Error (MSE): {mse_dr:.4f}\n  R-squared (R2) Score: {r2_dr:.4f}")
    figs.append(plot_rq2(y_test_dr, preds_dr, out_path='figures/RQ2_Fig1.pdf'))

    # RQ3 and RQ4
    df_regression['DANGCEM_Daily_Returns'] = df_regression['Price'].pct_change()
    df_regression.dropna(inplace=True)
    df_regression['Absolute_Daily_Change'] = df_regression['DANGCEM_Daily_Returns'].abs()
    df_regression['Volume_Category'] = pd.qcut(df_regression['Vol.'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')

    figs.append(plot_rq3(df_regression, out_path='figures/RQ3_Fig1.pdf'))
    rq4_a, rq4_b = plot_rq4(df_regression, out_path1='figures/RQ4_Fig1.pdf', out_path2='figures/RQ4_Fig2.pdf')
    figs.extend([rq4_a, rq4_b])

    figs = [f.strip() for f in figs if f and isinstance(f, str)]
    if figs:
        print("Figures plotted:", ", ".join(figs))
    else:
        print("No figures were created.")

    print(f"Pipeline complete. {len(figs)} figure(s) generated and saved to 'figures/'. Models saved to 'models/'. Processed data saved to 'data/processed/dangcem_cleanedx2.csv'.")


if __name__ == "__main__":
    main()