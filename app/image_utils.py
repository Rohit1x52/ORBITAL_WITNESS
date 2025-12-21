import cv2
import numpy as np

def detect_changes(before_image: np.ndarray, after_image: np.ndarray) -> np.ndarray:
    print("Detecting changes between images...")
    diff = cv2.absdiff(before_image, after_image)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, threshold_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    std_size = (256, 256)
    resized_diff = cv2.resize(threshold_diff, std_size, interpolation=cv2.INTER_AREA)
    final_diff_map = cv2.cvtColor(resized_diff, cv2.COLOR_GRAY2BGR)
    print("Change detection complete.")
    return final_diff_map

def preprocess_image(image: np.ndarray) -> np.ndarray:
    print("Preprocessing image for classification...")
    std_size = (224, 224)
    resized_image = cv2.resize(image, std_size, interpolation=cv2.INTER_AREA)
    return resized_image

