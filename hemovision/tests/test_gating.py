
import pandas as pd
from hemovision.fusion.gating import GatingModel, GatingConfig

def test_gating_weights_sum_to_one():
    df = pd.DataFrame({
        "y_pred_nail":[12.0, 13.5, 14.0],
        "y_pred_conj":[12.5, 13.0, 14.5],
        "y":[12.2, 13.1, 14.3]
    })
    gm = GatingModel(GatingConfig(epochs=2, hidden=4))  # short train
    gm.fit(df)
    out = gm.predict_weights(df)
    s = (out["w_nail"] + out["w_conj"]).round(6)
    assert (s == 1.0).all()
