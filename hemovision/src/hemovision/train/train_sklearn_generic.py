
import numpy as np
import pandas as pd
from typing import Callable
from tqdm import tqdm
from ..data.manifest import load_manifest, subjectwise_split
from ..preprocess.io import read_image
from ..models.sklearn_regressor import SklearnRegressor, SklearnRegressorConfig
from ..eval.metrics import mae
import os, json

def extract_features(df: pd.DataFrame, image_root: str, feature_fn: Callable) -> np.ndarray:
    feats = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img = read_image(image_root, row["image_path"])
        f = feature_fn(img)
        feats.append(np.asarray(f, dtype=float).ravel())
    return np.vstack(feats)

def train_eval_generic(manifest: str, image_root: str, out_dir: str, feature_fn: Callable, n_splits: int = 5, fold: int = 0):
    df = load_manifest(manifest)
    train_df, val_df = subjectwise_split(df, n_splits=n_splits, fold=fold)

    X_train = extract_features(train_df, image_root, feature_fn)
    y_train = train_df["hb_value"].values

    X_val = extract_features(val_df, image_root, feature_fn)
    y_val = val_df["hb_value"].values

    model = SklearnRegressor(SklearnRegressorConfig())
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    val_mae = mae(y_val, pred)

    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.joblib"))
    pd.DataFrame({"image_path": val_df["image_path"], "subject_id": val_df.get("subject_id"), "y": y_val, "y_pred": pred}).to_csv(
        os.path.join(out_dir, "val_predictions.csv"), index=False
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"val_mae": float(val_mae)}, f, indent=2)

    return {"val_mae": float(val_mae)}
