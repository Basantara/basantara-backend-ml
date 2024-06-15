import numpy as np

from PIL import Image
from io import BytesIO

def load_image(data):
    image = Image.open(BytesIO(data))
    return image