
import argparse
from hemovision.train.train_sklearn import train_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    metrics = train_eval(args.manifest, args.image_root, args.out_dir, args.n_splits, args.fold)
    print("Val MAE:", metrics["val_mae"])

if __name__ == "__main__":
    main()
