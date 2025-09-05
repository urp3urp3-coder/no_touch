
import argparse, os
import pandas as pd
from hemovision.fusion.gating import GatingModel, GatingConfig

def merge_two(nail_csv: str, conj_csv: str):
    dn = pd.read_csv(nail_csv).rename(columns={"y_pred":"y_pred_nail"})
    dc = pd.read_csv(conj_csv).rename(columns={"y_pred":"y_pred_conj"})
    key = "image_path" if "image_path" in dn.columns and "image_path" in dc.columns else "subject_id"
    merged = pd.merge(dn, dc, on=key, suffixes=("_nail","_conj"))
    if "y" not in merged.columns:
        ycol = "y_nail" if "y_nail" in merged.columns else ("y_conj" if "y_conj" in merged.columns else None)
        if ycol is None:
            raise ValueError("Ground-truth column 'y' not found in either CSV")
        merged = merged.rename(columns={ycol:"y"})
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nail", required=True, help="val_predictions.csv from nail")
    ap.add_argument("--conj", required=True, help="val_predictions.csv from conjunctiva")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    merged = merge_two(args.nail, args.conj)
    cfg = GatingConfig(hidden=args.hidden, lr=args.lr, epochs=args.epochs)
    model = GatingModel(cfg)
    metrics = model.fit(merged)
    os.makedirs(args.out_dir, exist_ok=True)
    model.save(os.path.join(args.out_dir, "gating.joblib"))
    print("Train MAE (gating fused):", metrics["train_mae"])

if __name__ == "__main__":
    main()
