
import pandas as pd

def fuse_weighted(nail_csv: str, conj_csv: str, out_csv: str, w_nail: float = 0.5, w_conj: float = 0.5):
    dn = pd.read_csv(nail_csv).rename(columns={"y_pred":"y_pred_nail"})
    dc = pd.read_csv(conj_csv).rename(columns={"y_pred":"y_pred_conj"})
    key = "image_path" if "image_path" in dn.columns and "image_path" in dc.columns else "subject_id"
    merged = pd.merge(dn, dc, on=key, suffixes=("_nail","_conj"))
    merged["y_fused"] = w_nail * merged["y_pred_nail"] + w_conj * merged["y_pred_conj"]
    merged.to_csv(out_csv, index=False)
    return out_csv
