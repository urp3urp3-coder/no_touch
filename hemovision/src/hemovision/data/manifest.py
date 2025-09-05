
import pandas as pd
from typing import Tuple
from sklearn.model_selection import GroupKFold

REQUIRED_COLUMNS = ["image_path", "subject_id", "hb_value"]

def load_manifest(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Manifest must contain column: {c}")
    return df

def subjectwise_split(df: pd.DataFrame, n_splits: int = 5, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/val split for given fold using subject-wise grouping."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["subject_id"].values
    splits = list(gkf.split(df, groups=groups))
    train_idx, val_idx = splits[fold]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
