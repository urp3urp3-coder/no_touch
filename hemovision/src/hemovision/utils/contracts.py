
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Prediction(BaseModel):
    hb_mean: float
    hb_std: Optional[float] = None
    anemia_prob: Optional[float] = None
    quality_flags: Optional[Dict[str, Any]] = None
