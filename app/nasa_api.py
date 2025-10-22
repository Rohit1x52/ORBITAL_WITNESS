import os
import requests
import numpy as np
import cv2
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()

# Fetch the API key from the environment
NASA_API_KEY = os.getenv("NASA_API_KEY")
if not NASA_API_KEY:
    # This error will stop the app if the key is missing, which is good.
    raise ValueError("NASA_API_KEY not found. Please set it in your .env file.")

# The base URL for the NASA Landsat 8 imagery API
BASE_URL = "https://api.nasa.gov/planetary/earth/imagery"

def fetch_imagery(location: tuple, date: str) -> np.ndarray:
    """
    Fetches a single Landsat 8 image from NASA's Earth API for a specific date.

    Args:
        location (tuple): The (latitude, longitude) of the area.
        date (str): The date of the image in 'YYYY-MM-DD' format.

    Returns:
        The fetched image as a NumPy array (OpenCV format).
    
    Raises:
        Exception: If the API call fails or no image is found.
    """
    lat, lon = location
    print(f"Fetching imagery for {location} on {date}...")

    params = {
        "lon": lon,
        "lat": lat,
        "date": date,
        "dim": 0.15,  # Sets the width and height of the image in degrees (0.15 is a good default)
        "api_key": NASA_API_KEY,
    }

    try:
        # Make the API request
        response = requests.get(BASE_URL, params=params)
        
        # This will raise an error if the request failed (e.g., 404, 500)
        response.raise_for_status()

        # Check if the content type is an image, not an error message
        if 'image' not in response.headers.get('Content-Type', ''):
             raise Exception(f"API did not return an image. Response: {response.text}")

        # Convert the raw image content (bytes) into a NumPy array
        image_array = np.frombuffer(response.content, np.uint8)
        
        # Decode the NumPy array into a full-color image (BGR format for OpenCV)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception("Failed to decode image from API response.")

        print(f"Successfully fetched image for {date}.")
        return image

    except requests.exceptions.HTTPError as http_err:
        # Handle specific API errors, like no image for that date
        if response.status_code == 404:
            raise Exception(f"No satellite imagery found for the specified date and location: {date}. Try a different date.")
        # Handle other HTTP errors
        raise Exception(f"HTTP error occurred: {http_err} - {response.text}")
    except Exception as e:
        # Catch any other errors (network issues, etc.)
        raise Exception(f"An error occurred while fetching imagery: {e}")