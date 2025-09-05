
import numpy as np

def calculate_features_conj(img):
    """간단 색상 통계 기반 특징 (placeholder).
    추후 결막 ROI/혈관 마스크 기반 특징으로 교체하세요.
    """
    x = img.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    R, G, B = x[...,0], x[...,1], x[...,2]
    feats = [
        R.mean(), G.mean(), B.mean(),
        R.std(),  G.std(),  B.std(),
        (R-G).mean(), (R-B).mean(),
        (R/(R+G+B+1e-6)).mean()
    ]
    return np.array(feats, dtype=np.float32)
