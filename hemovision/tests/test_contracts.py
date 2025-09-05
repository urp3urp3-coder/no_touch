
from hemovision.utils.contracts import Prediction

def test_prediction_schema():
    p = Prediction(hb_mean=12.3, hb_std=0.4, anemia_prob=0.2, quality_flags={"blur": False})
    assert p.hb_mean == 12.3
