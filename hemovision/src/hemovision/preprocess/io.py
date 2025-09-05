
from skimage import io as skio
import numpy as np
import os

def read_image(image_root: str, relative_path: str):
    path = os.path.join(image_root, relative_path)
    img = skio.imread(path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    return img
