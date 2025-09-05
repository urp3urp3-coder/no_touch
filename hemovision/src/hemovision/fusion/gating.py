
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import joblib
import numpy as np

FEATURE_CANDIDATES = [
    "y_pred_nail","y_pred_conj",
    "hb_std_nail","hb_std_conj",
    "anemia_prob_nail","anemia_prob_conj"
]

def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
    if not cols:
        cols = ["y_pred_nail","y_pred_conj"]
    X = df[cols].values.astype("float32")
    return X

@dataclass
class GatingConfig:
    hidden: int = 16
    lr: float = 1e-3
    epochs: int = 100
    weight_decay: float = 0.0
    seed: int = 42

class GatingMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        logits = self.net(x)
        w = torch.softmax(logits, dim=-1)
        return w

class GatingModel:
    def __init__(self, cfg: GatingConfig):
        self.cfg = cfg
        self.model: Optional[GatingMLP] = None

    def fit(self, merged: pd.DataFrame) -> dict:
        torch.manual_seed(self.cfg.seed)
        X = _prepare_features(merged)
        y_nail = merged["y_pred_nail"].values.astype("float32")
        y_conj = merged["y_pred_conj"].values.astype("float32")
        y_true = merged["y"].values.astype("float32")

        X_t = torch.from_numpy(X)
        y_true_t = torch.from_numpy(y_true)
        yn_t = torch.from_numpy(y_nail)
        yc_t = torch.from_numpy(y_conj)

        self.model = GatingMLP(in_dim=X.shape[1], hidden=self.cfg.hidden)
        opt = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        loss_fn = nn.L1Loss()

        self.model.train()
        for _ in range(self.cfg.epochs):
            opt.zero_grad()
            w = self.model(X_t)  # [N,2]
            y_hat = w[:,0]*yn_t + w[:,1]*yc_t
            loss = loss_fn(y_hat, y_true_t)
            loss.backward()
            opt.step()

        with torch.no_grad():
            w = self.model(X_t)
            y_hat = w[:,0]*yn_t + w[:,1]*yc_t
            mae = torch.mean(torch.abs(y_hat - y_true_t)).item()
        return {"train_mae": mae}

    def predict_weights(self, merged: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None, "Model not fitted or loaded"
        self.model.eval()
        X = _prepare_features(merged)
        with torch.no_grad():
            w = self.model(torch.from_numpy(X)).numpy()
        out = merged.copy()
        out["w_nail"] = w[:,0]
        out["w_conj"] = w[:,1]
        return out

    def save(self, path: str):
        assert self.model is not None
        obj = {"state_dict": self.model.state_dict(), "cfg": self.cfg, "in_dim": list(self.model.net.children())[0].in_features}
        joblib.dump(obj, path)

    @classmethod
    def load(cls, path: str) -> "GatingModel":
        obj = joblib.load(path)
        inst = cls(obj["cfg"])
        inst.model = GatingMLP(in_dim=obj["in_dim"], hidden=obj["cfg"].hidden)
        inst.model.load_state_dict(obj["state_dict"])
        inst.model.eval()
        return inst
