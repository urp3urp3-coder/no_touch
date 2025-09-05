
import argparse
import pandas as pd
from hemovision.fusion.weighted import fuse_weighted
from hemovision.fusion.gating import GatingModel

def merge_for_inference(nail_csv: str, conj_csv: str):
    dn = pd.read_csv(nail_csv).rename(columns={"y_pred":"y_pred_nail"})
    dc = pd.read_csv(conj_csv).rename(columns={"y_pred":"y_pred_conj"})
    key = "image_path" if "image_path" in dn.columns and "image_path" in dc.columns else "subject_id"
    merged = pd.merge(dn, dc, on=key, suffixes=("_nail","_conj"))
    return merged, key

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nail", required=True, help="CSV with y_pred for nail")
    ap.add_argument("--conj", required=True, help="CSV with y_pred for conjunctiva")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["weighted","gating"], default="weighted")
    ap.add_argument("--w-nail", type=float, default=0.5)
    ap.add_argument("--w-conj", type=float, default=0.5)
    ap.add_argument("--gating-model", default=None, help="Path to gating.joblib for --method gating")
    args = ap.parse_args()

    if args.method == "weighted":
        fuse_weighted(args.nail, args.conj, args.out, args.w_nail, args.w_conj)
    else:
        assert args.gating_model, "Provide --gating-model when method=gating"
        merged, key = merge_for_inference(args.nail, args.conj)
        model = GatingModel.load(args.gating_model)
        out = model.predict_weights(merged)
        out["y_fused"] = out["w_nail"]*out["y_pred_nail"] + out["w_conj"]*out["y_pred_conj"]
        out[[key, "y_pred_nail","y_pred_conj","w_nail","w_conj","y_fused"]].to_csv(args.out, index=False)
        print(f"Saved fused predictions to {args.out}")

if __name__ == "__main__":
    main()
