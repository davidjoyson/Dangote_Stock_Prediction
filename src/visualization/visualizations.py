import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_time_series(df: pd.DataFrame, out_path: str = 'figures/TimeSeries_Fig1.pdf'):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'], label='Closing Price', alpha=0.7, color='blue')
    plt.plot(df.index, df['MA_7'], label='One Week Moving Average', color='orange', linestyle='--')
    plt.plot(df.index, df['MA_30'], label='One Month Moving Average', color='green', linestyle='--')
    plt.title('Dangote Cement Stock Price History & Moving Averages Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_correlation(df: pd.DataFrame, out_path: str = 'figures/Correlation_Fig1.pdf'):
    numerical_cols = df.select_dtypes(include='number').columns
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Stock price Prediction Features')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_model_comparison(y_test, predictions: dict, out_path: str = 'figures/Model_comparison_Fig1.pdf'):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual Price', color='black', linewidth=2)
    for name, preds in predictions.items():
        if name in ["Linear Regression", "Random Forest", "SVR"]:
            plt.plot(y_test.index, preds, label=name, linestyle='--', alpha=0.8)
    plt.title('Model Comparison: Predicted vs Actual Next Day Price')
    plt.xlabel('Time Index')
    plt.ylabel('Standardized Price')
    plt.legend()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_rq1(y_test, pred_lr, out_path: str = 'figures/RQ1_Fig1.pdf'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
    plt.plot(y_test.index, pred_lr, label='Predicted Price', color='orange', linestyle='--')
    plt.title('Stock Price Forecasting: Linear Regression')
    plt.xlabel('Date')
    plt.ylabel('Price (Standardized)')
    plt.legend()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_rq2(y_test, preds, out_path: str = 'figures/RQ2_Fig1.pdf'):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label='Actual Daily Returns', color='blue')
    plt.plot(y_test.index, preds, label='Predicted Daily Returns', color='red', linestyle='--')
    plt.title('Multiple Linear Regression: Actual vs. Predicted Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('DANGCEM Daily Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_rq3(df_regression, out_path: str = 'figures/RQ3_Fig1.pdf'):
    df = df_regression.copy()
    df['Month'] = df.index.month
    months_map = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    df['Month'] = df['Month'].map(months_map)
    monthly_avg_returns = df.groupby('Month')['DANGCEM_Daily_Returns'].mean().reindex([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    plt.figure(figsize=(12, 6))
    sns.barplot(x=monthly_avg_returns.index, y=monthly_avg_returns.values, hue=monthly_avg_returns.index, palette='viridis', dodge=False)
    plt.title('Monthly Average DANGCEM Daily Returns')
    plt.xlabel('Month')
    plt.ylabel('Average Daily Return')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_rq4(df_regression, out_path1: str = 'figures/RQ4_Fig1.pdf', out_path2: str = 'figures/RQ4_Fig2.pdf'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Vol.', y='Daily_Range', data=df_regression, alpha=0.6)
    plt.title('Trading Volume vs. Daily Price Range')
    plt.xlabel('Trading Volume (Standardized)')
    plt.ylabel('Daily Price Range (Standardized)')
    plt.grid(True)
    plt.savefig(out_path1, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Volume_Category', y='Absolute_Daily_Change', data=df_regression, palette='viridis', order=['Low', 'Medium', 'High'], hue='Volume_Category', legend=False)
    plt.title('Distribution of Absolute Daily Price Change by Volume Category')
    plt.xlabel('Volume Category')
    plt.ylabel('Absolute Daily Price Change (Standardized)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(out_path2, dpi=300)
    plt.close()
    return out_path1, out_path2
