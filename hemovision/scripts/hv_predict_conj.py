
import argparse, os
import pandas as pd
import numpy as np
from hemovision.models.sklearn_regressor import SklearnRegressor
from hemovision.preprocess.io import read_image
from hemovision.preprocess.conj_features import calculate_features_conj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = SklearnRegressor.load(args.model)
    df = pd.read_csv(args.manifest)
    X = []
    for _, row in df.iterrows():
        img = read_image(args.image_root, row["image_path"])
        X.append(np.asarray(calculate_features_conj(img)).ravel())
    X = np.vstack(X)
    pred = model.predict(X)
    out = df.copy()
    out["y_pred"] = pred
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

if __name__ == "__main__":
    main()
