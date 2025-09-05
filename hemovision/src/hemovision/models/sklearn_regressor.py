
from dataclasses import dataclass
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

@dataclass
class SklearnRegressorConfig:
    n_estimators: int = 200
    random_state: int = 42

class SklearnRegressor:
    def __init__(self, cfg: SklearnRegressorConfig = SklearnRegressorConfig()):
        self.cfg = cfg
        self.model = RandomForestRegressor(
            n_estimators=cfg.n_estimators, random_state=cfg.random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump({"cfg": self.cfg, "model": self.model}, path)

    @classmethod
    def load(cls, path: str) -> "SklearnRegressor":
        obj = joblib.load(path)
        inst = cls(obj["cfg"])
        inst.model = obj["model"]
        return inst
