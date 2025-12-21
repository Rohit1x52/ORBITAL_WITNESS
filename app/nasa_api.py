import os
import requests
import numpy as np
import cv2
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
NASA_API_KEY = os.getenv("NASA_API_KEY")
BASE_URL = "https://api.nasa.gov/planetary/earth/imagery"
CACHE_DIR = "./imagery_cache"

os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_imagery_smart(location: tuple, date: str, search_window_days=7) -> np.ndarray:
    lat, lon = location
    target_date_obj = datetime.strptime(date, "%Y-%m-%d")

    cache_path = f"{CACHE_DIR}/{lat}_{lon}_{date}.png"
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        return cv2.imread(cache_path)

    for i in range(search_window_days + 1):
        for offset in (0, i, -i):
            if i == 0 and offset != 0: continue
            
            check_date = (target_date_obj + timedelta(days=offset)).strftime("%Y-%m-%d")
            print(f"Checking NASA API for {check_date}...")

            params = {
                "lon": lon, "lat": lat, "date": check_date,
                "dim": 0.15, "api_key": NASA_API_KEY,
            }

            response = requests.get(BASE_URL, params=params)
            
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                image_array = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                avg_brightness = np.mean(image)
                if avg_brightness > 220:
                    print(f"Skipping {check_date}: Too cloudy/bright.")
                    continue

                cv2.imwrite(f"{CACHE_DIR}/{lat}_{lon}_{check_date}.png", image)
                print(f"Success! Found valid imagery on {check_date}")
                return image
            
    raise Exception(f"No clear imagery found near {date} for location {location}")