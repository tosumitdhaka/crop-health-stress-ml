"""Loading and splitting the dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, POSITIVE_LABEL, SEED, TARGET_COL, TEST_SIZE


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def make_target(df: pd.DataFrame) -> pd.Series:
    """1 = Stressed, 0 = Healthy."""
    return (df[TARGET_COL] == POSITIVE_LABEL).astype(int).rename("stressed")


def split(df: pd.DataFrame):
    """Stratified 80/20 split. Returns (train_df, test_df) with all columns kept
    so grid coordinates stay available for spatial analysis."""
    y = make_target(df)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    return train_df.copy(), test_df.copy()
