import pandas as pd


def load_raw(path: str = 'data/raw/Dangote Cement Stock Price History.csv') -> pd.DataFrame:
    """Load raw stock price history CSV and return a DataFrame."""
    df = pd.read_csv(path)
    return df
