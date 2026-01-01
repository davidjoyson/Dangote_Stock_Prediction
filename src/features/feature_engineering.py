import pandas as pd
from typing import Iterable


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic technical features: Prev_Close, MA_7, MA_30, Daily Return, Daily_Range.

    Returns a new DataFrame with the features added (does not scale).
    """
    df = df.copy()
    df['Prev_Close'] = df['Price'].shift(1)
    df['MA_7'] = df['Price'].rolling(window=7).mean()
    df['MA_30'] = df['Price'].rolling(window=30).mean()
    df['Daily Return'] = df['Price'].pct_change()
    df['Daily_Range'] = df['High'] - df['Low']
    return df


def add_lag_features(df: pd.DataFrame, lags: Iterable[int] = (1, 2, 3)) -> pd.DataFrame:
    """Add lag features (Lag_1, Lag_2, ...) for price-based models."""
    df = df.copy()
    for lag in lags:
        df[f'Lag_{lag}'] = df['Price'].shift(lag)
    return df
