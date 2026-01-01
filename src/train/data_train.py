import numpy as np
import pandas as pd
import os
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


def prepare_lag_features(df: pd.DataFrame, lags=(1,2,3)):
    # Use features helper to add lag columns (keeps API stable)
    from features import add_lag_features
    df_with_lags = add_lag_features(df, lags=lags).copy()
    df_with_lags.dropna(inplace=True)
    X = df_with_lags[[f'Lag_{lag}' for lag in lags]]
    y = df_with_lags['Price']
    return X, y


def train_basic_lr(X, y, save_dir: str = 'models'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ensure models directory exists and save the trained model
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(save_dir) / 'basic_lr.joblib'
    dump(model, model_path)
    print(f"Saved model: {model_path}")

    preds = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, preds


def prepare_next_price_training(df: pd.DataFrame, features=None):
    df = df.copy()
    df['Next_Price'] = df['Price'].shift(-1)
    df_model = df.dropna().copy()
    if features is None:
        features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'MA_7', 'MA_30']
    X = df_model[features]
    y = df_model['Next_Price']
    train_size = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    return X_train, X_test, y_train, y_test


def train_models_dict(X_train, y_train, X_test, y_test, save_dir: str = 'models'):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    results = []
    predictions = {}
    saved_paths = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        # Save the fitted model
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        model_path = Path(save_dir) / f"{name.replace(' ', '_')}.joblib"
        dump(model, model_path)
        saved_paths.append(str(model_path))

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        })

    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

    if saved_paths:
        print("Saved models:", ", ".join(saved_paths))

    return models, predictions, results_df


def train_daily_returns_model(df: pd.DataFrame):
    df = df.copy()
    df['DANGCEM_Daily_Returns'] = df['Price'].pct_change()
    df.dropna(inplace=True)
    y = df['DANGCEM_Daily_Returns']
    X = df[['Price', 'Open', 'Prev_Close', 'MA_7', 'MA_30', 'Daily_Range']]
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained daily returns model
    Path('models').mkdir(parents=True, exist_ok=True)
    model_path = Path('models') / 'daily_returns_lr.joblib'
    dump(model, model_path)
    print(f"Saved model: {model_path}")

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return model, X_train, X_test, y_train, y_test, preds, mse, r2
