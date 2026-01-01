import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_volume(volume_str):
    if isinstance(volume_str, str):
        volume_str = volume_str.replace(',', '')
        if 'K' in volume_str:
            return float(volume_str.replace('K', '')) * 1000
        elif 'M' in volume_str:
            return float(volume_str.replace('M', '')) * 1000000
        else:
            return float(volume_str)
    return volume_str


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw DataFrame and return scaled, feature-engineered DataFrame."""
    df = df.copy()
    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Basic cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Clean columns
    df['Vol.'] = df['Vol.'].apply(clean_volume)
    df['Change %'] = df['Change %'].str.replace('%', '', regex=False).astype(float)

    # Sort and set index
    df.sort_values(by='Date', ascending=True, inplace=True)
    df.set_index('Date', inplace=True)

    # Feature engineering (moved to features helper)
    from features import add_technical_features
    df = add_technical_features(df)

    # Drop rows with NaNs produced by feature creation
    df.dropna(inplace=True)

    # Scaling
    features_to_scale = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %',
                         'Prev_Close', 'MA_7', 'MA_30', 'Daily Return', 'Daily_Range']
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df_scaled


def save_processed(df: pd.DataFrame, path: str = 'data/processed/dangcem_cleanedx2.csv') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)
