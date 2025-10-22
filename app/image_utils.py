import cv2
import numpy as np

# ... existing detect_changes function ...
def detect_changes(before_image: np.ndarray, after_image: np.ndarray) -> np.ndarray:
    """
    Compares two images to find and highlight significant differences.

    Args:
        before_image (np.ndarray): The 'before' image.
        after_image (np.ndarray): The 'after' image.

    Returns:
        A binary 'difference map' image (black and white) where
        white pixels represent areas of significant change.
    """
    print("Detecting changes between images...")
    
    # Calculate the absolute difference between the two images
    diff = cv2.absdiff(before_image, after_image)
    
    # Convert the difference to grayscale
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to get a binary image of significant changes
    _, threshold_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Resize diff map to a standard size for the classifier
    std_size = (256, 256)
    resized_diff = cv2.resize(threshold_diff, std_size, interpolation=cv2.INTER_AREA)
    
    # Convert back to BGR for models that expect 3 channels
    final_diff_map = cv2.cvtColor(resized_diff, cv2.COLOR_GRAY2BGR)
    
    print("Change detection complete.")
    return final_diff_map


# --- ADD THIS NEW FUNCTION ---
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Prepares an image for the classification model.
    (e.g., resize, normalize, etc.)
    
    NOTE: This is a simple placeholder. A real model might
    require normalization like (image / 255.0).
    """
    print("Preprocessing image for classification...")
    # Most vision models expect a fixed size.
    std_size = (224, 224)
    resized_image = cv2.resize(image, std_size, interpolation=cv2.INTER_AREA)
    return resized_image

