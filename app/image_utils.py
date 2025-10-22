from PIL import Image
import numpy as np

def preprocess_image(image_np, size=(224, 224)):
    image = Image.fromarray(image_np)
    image = image.resize(size)
    return np.array(image)