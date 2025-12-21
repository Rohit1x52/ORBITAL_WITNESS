import random

def classify_image(image_np):
    labels = ["deforestation", "urban", "flood", "normal", "wildfire", "volcanic_eruption", "bombardment"]
    label = random.choice(labels)
    confidence = round(random.uniform(0.6, 0.95), 2)
    return {"label": label, "confidence": confidence}