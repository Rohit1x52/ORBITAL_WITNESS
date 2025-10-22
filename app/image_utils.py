import cv2
import numpy as np

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
    
    # Resize 'after' image to match 'before' image dimensions.
    height, width, _ = before_image.shape
    after_image_resized = cv2.resize(after_image, (width, height))

    # Convert both images to grayscale for easier comparison
    gray_before = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_image_resized, cv2.COLOR_BGR2GRAY)

    # Apply a slight blur to reduce noise (e.g., sensor noise, minor shadow differences)
    gray_before = cv2.GaussianBlur(gray_before, (5, 5), 0)
    gray_after = cv2.GaussianBlur(gray_after, (5, 5), 0)

    # --- 2. Change Detection ---

    # Calculate the absolute difference between the two blurred grayscale images
    diff = cv2.absdiff(gray_before, gray_after)
    threshold_value = 30
    _, threshold_diff = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # --- 3. Noise Reduction (Morphological Operations) ---

    kernel = np.ones((5, 5), np.uint8)
    diff_map = cv2.dilate(threshold_diff, kernel, iterations=2)

    print("Change detection complete.")
    return diff_map
