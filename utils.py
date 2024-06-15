import numpy as np

from PIL import Image
from io import BytesIO

def load_image_into_numpy_array(data):
    image = Image.open(BytesIO(data))
    return image