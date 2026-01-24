import os
import requests
import numpy as np
import cv2
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

logger = logging.getLogger(__name__)


class ImageQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CLOUDY = "cloudy"


@dataclass
class ImageryConfig:
    cache_dir: str = "./imagery_cache"
    api_timeout: int = 30
    max_retries: int = 3
    search_window_days: int = 7
    min_brightness: float = 10.0
    max_brightness: float = 220.0
    min_contrast: float = 20.0
    cloud_threshold: float = 220.0
    dimension: float = 0.15
    preferred_resolution: Tuple[int, int] = (512, 512)
    cache_metadata: bool = True
    parallel_requests: bool = False
    max_workers: int = 4


@dataclass
class ImageMetadata:
    location: Tuple[float, float]
    date: str
    actual_date: str
    quality: str
    brightness: float
    contrast: float
    file_size: int
    resolution: Tuple[int, int]
    cloud_score: float
    fetch_timestamp: str
    source: str = "nasa_earth_api"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class ImageQualityAnalyzer:
    def __init__(self, config: ImageryConfig):
        self.config = config
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(np.mean(gray))
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(np.std(gray))
    
    def detect_clouds(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        bright_pixels = np.sum(gray > self.config.cloud_threshold)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        return float(bright_pixels / total_pixels * 100)
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))
    
    def assess_quality(self, image: np.ndarray) -> Tuple[ImageQuality, Dict[str, float]]:
        metrics = {
            'brightness': self.calculate_brightness(image),
            'contrast': self.calculate_contrast(image),
            'cloud_coverage': self.detect_clouds(image),
            'sharpness': self.calculate_sharpness(image)
        }
        
        if metrics['cloud_coverage'] > 30:
            quality = ImageQuality.CLOUDY
        elif metrics['brightness'] > self.config.max_brightness:
            quality = ImageQuality.CLOUDY
        elif metrics['brightness'] < self.config.min_brightness:
            quality = ImageQuality.POOR
        elif metrics['contrast'] < self.config.min_contrast:
            quality = ImageQuality.POOR
        elif metrics['sharpness'] > 500 and metrics['cloud_coverage'] < 10:
            quality = ImageQuality.EXCELLENT
        elif metrics['sharpness'] > 300 and metrics['cloud_coverage'] < 20:
            quality = ImageQuality.GOOD
        else:
            quality = ImageQuality.FAIR
        
        return quality, metrics


class CacheManager:
    def __init__(self, cache_dir: str, enable_metadata: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.cache_dir / "metadata"
        self.enable_metadata = enable_metadata
        
        if enable_metadata:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_cache_key(self, lat: float, lon: float, date: str) -> str:
        key_string = f"{lat}_{lon}_{date}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, lat: float, lon: float, date: str) -> Path:
        cache_key = self._generate_cache_key(lat, lon, date)
        return self.cache_dir / f"{cache_key}.png"
    
    def _get_metadata_path(self, lat: float, lon: float, date: str) -> Path:
        cache_key = self._generate_cache_key(lat, lon, date)
        return self.metadata_dir / f"{cache_key}.json"
    
    def exists(self, lat: float, lon: float, date: str) -> bool:
        return self._get_cache_path(lat, lon, date).exists()
    
    def save(
        self, 
        image: np.ndarray, 
        lat: float, 
        lon: float, 
        date: str,
        metadata: Optional[ImageMetadata] = None
    ):
        cache_path = self._get_cache_path(lat, lon, date)
        cv2.imwrite(str(cache_path), image)
        logger.info(f"Cached image: {cache_path}")
        
        if self.enable_metadata and metadata:
            metadata_path = self._get_metadata_path(lat, lon, date)
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
    
    def load(
        self, 
        lat: float, 
        lon: float, 
        date: str,
        load_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[ImageMetadata]]]:
        cache_path = self._get_cache_path(lat, lon, date)
        
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")
        
        image = cv2.imread(str(cache_path))
        logger.info(f"Loaded from cache: {cache_path}")
        
        if load_metadata and self.enable_metadata:
            metadata_path = self._get_metadata_path(lat, lon, date)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                metadata = ImageMetadata.from_dict(metadata_dict)
                return image, metadata
        
        return image
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        if older_than_days:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            for file_path in self.cache_dir.glob("*.png"):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_time:
                    file_path.unlink()
                    logger.info(f"Deleted old cache: {file_path}")
        else:
            for file_path in self.cache_dir.glob("*.png"):
                file_path.unlink()
            logger.info("Cache cleared")


class NASAEarthImageryAPI:
    BASE_URL = "https://api.nasa.gov/planetary/earth/imagery"
    ASSETS_URL = "https://api.nasa.gov/planetary/earth/assets"
    
    def __init__(self, config: Optional[ImageryConfig] = None):
        load_dotenv()
        self.api_key = os.getenv("NASA_API_KEY")
        
        if not self.api_key:
            raise ValueError("NASA_API_KEY not found in environment variables")
        
        self.config = config or ImageryConfig()
        self.cache_manager = CacheManager(
            self.config.cache_dir, 
            self.config.cache_metadata
        )
        self.quality_analyzer = ImageQualityAnalyzer(self.config)
        self.session = requests.Session()
    
    def _make_request(
        self, 
        url: str, 
        params: Dict,
        retry_count: int = 0
    ) -> requests.Response:
        try:
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.config.api_timeout
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if retry_count < self.config.max_retries:
                logger.warning(
                    f"Request failed (attempt {retry_count + 1}/{self.config.max_retries}): {e}"
                )
                return self._make_request(url, params, retry_count + 1)
            else:
                logger.error(f"Request failed after {self.config.max_retries} retries: {e}")
                raise
    
    def _get_date_range(
        self, 
        target_date: datetime, 
        window_days: int
    ) -> List[Tuple[str, int]]:
        dates = []
        for i in range(window_days + 1):
            for offset in (0, i, -i):
                if i == 0 and offset != 0:
                    continue
                
                check_date = target_date + timedelta(days=offset)
                date_str = check_date.strftime("%Y-%m-%d")
                dates.append((date_str, abs(offset)))
        
        return sorted(set(dates), key=lambda x: x[1])
    
    def fetch_single_date(
        self,
        lat: float,
        lon: float,
        date: str,
        quality_check: bool = True
    ) -> Optional[Tuple[np.ndarray, ImageMetadata]]:
        params = {
            "lon": lon,
            "lat": lat,
            "date": date,
            "dim": self.config.dimension,
            "api_key": self.api_key,
        }
        
        try:
            response = self._make_request(self.BASE_URL, params)
            
            if 'image' not in response.headers.get('Content-Type', ''):
                logger.warning(f"Invalid content type for {date}")
                return None
            
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.warning(f"Failed to decode image for {date}")
                return None
            
            if self.config.preferred_resolution:
                image = cv2.resize(
                    image, 
                    self.config.preferred_resolution,
                    interpolation=cv2.INTER_LANCZOS4
                )
            
            quality, metrics = self.quality_analyzer.assess_quality(image)
            
            if quality_check and quality == ImageQuality.CLOUDY:
                logger.info(f"Skipping {date}: {quality.value} (cloud coverage: {metrics['cloud_coverage']:.1f}%)")
                return None
            
            metadata = ImageMetadata(
                location=(lat, lon),
                date=date,
                actual_date=date,
                quality=quality.value,
                brightness=metrics['brightness'],
                contrast=metrics['contrast'],
                file_size=len(response.content),
                resolution=image.shape[:2],
                cloud_score=metrics['cloud_coverage'],
                fetch_timestamp=datetime.now().isoformat()
            )
            
            logger.info(
                f"Fetched imagery for {date}: {quality.value} "
                f"(brightness: {metrics['brightness']:.1f}, "
                f"cloud: {metrics['cloud_coverage']:.1f}%)"
            )
            
            return image, metadata
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {date}: {e}")
            return None
    
    def fetch_imagery(
        self,
        location: Union[Tuple[float, float], str],
        date: str,
        use_cache: bool = True,
        quality_check: bool = True,
        return_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, ImageMetadata]]:
        if isinstance(location, str):
            lat, lon = map(float, location.split(','))
        else:
            lat, lon = location
        
        target_date = datetime.strptime(date, "%Y-%m-%d")
        
        if use_cache and self.cache_manager.exists(lat, lon, date):
            if return_metadata:
                return self.cache_manager.load(lat, lon, date, load_metadata=True)
            return self.cache_manager.load(lat, lon, date)
        
        date_range = self._get_date_range(target_date, self.config.search_window_days)
        
        logger.info(
            f"Searching for clear imagery around {date} "
            f"(±{self.config.search_window_days} days)"
        )
        
        best_image = None
        best_metadata = None
        best_quality_score = -1
        
        for check_date, offset in date_range:
            result = self.fetch_single_date(lat, lon, check_date, quality_check)
            
            if result is None:
                continue
            
            image, metadata = result
            
            quality_score = (
                metadata.contrast * 0.4 +
                (100 - metadata.cloud_score) * 0.4 +
                (255 - abs(metadata.brightness - 127)) * 0.2
            )
            
            if quality_score > best_quality_score:
                best_image = image
                best_metadata = metadata
                best_quality_score = quality_score
            
            if metadata.quality in [ImageQuality.EXCELLENT.value, ImageQuality.GOOD.value]:
                break
        
        if best_image is None:
            raise Exception(
                f"No clear imagery found near {date} for location ({lat}, {lon}) "
                f"within {self.config.search_window_days} days"
            )
        
        if use_cache:
            self.cache_manager.save(best_image, lat, lon, date, best_metadata)
        
        logger.info(
            f"Best imagery: {best_metadata.actual_date} "
            f"({best_metadata.quality}, score: {best_quality_score:.1f})"
        )
        
        if return_metadata:
            return best_image, best_metadata
        return best_image
    
    def fetch_imagery_batch(
        self,
        locations: List[Tuple[float, float]],
        date: str,
        use_cache: bool = True
    ) -> List[Tuple[np.ndarray, ImageMetadata]]:
        if not self.config.parallel_requests:
            return [
                self.fetch_imagery(loc, date, use_cache, return_metadata=True)
                for loc in locations
            ]
        
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_location = {
                executor.submit(
                    self.fetch_imagery, 
                    loc, 
                    date, 
                    use_cache,
                    return_metadata=True
                ): loc
                for loc in locations
            }
            
            for future in as_completed(future_to_location):
                location = future_to_location[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to fetch imagery for {location}: {e}")
        
        return results
    
    def get_available_dates(
        self,
        location: Union[Tuple[float, float], str],
        start_date: str,
        end_date: str
    ) -> List[str]:
        if isinstance(location, str):
            lat, lon = map(float, location.split(','))
        else:
            lat, lon = location
        
        params = {
            "lon": lon,
            "lat": lat,
            "begin": start_date,
            "end": end_date,
            "api_key": self.api_key
        }
        
        response = self._make_request(self.ASSETS_URL, params)
        data = response.json()
        
        dates = [item['date'] for item in data.get('results', [])]
        return sorted(dates)


def fetch_imagery_smart(
    location: Union[Tuple[float, float], str],
    date: str,
    search_window_days: int = 7,
    use_cache: bool = True
) -> np.ndarray:
    config = ImageryConfig(search_window_days=search_window_days)
    api = NASAEarthImageryAPI(config)
    return api.fetch_imagery(location, date, use_cache)


def fetch_imagery(
    location: Union[Tuple[float, float], str],
    date: str
) -> np.ndarray:
    return fetch_imagery_smart(location, date)


if __name__ == "__main__":
    config = ImageryConfig(
        search_window_days=10,
        cloud_threshold=210,
        cache_metadata=True,
        parallel_requests=False
    )
    
    api = NASAEarthImageryAPI(config)
    
    location = (34.0522, -118.2437)
    date = "2024-01-15"
    
    try:
        image, metadata = api.fetch_imagery(
            location, 
            date, 
            return_metadata=True
        )
        
        print(f"\nImagery Metadata:")
        print(f"  Date: {metadata.actual_date}")
        print(f"  Quality: {metadata.quality}")
        print(f"  Brightness: {metadata.brightness:.1f}")
        print(f"  Contrast: {metadata.contrast:.1f}")
        print(f"  Cloud Coverage: {metadata.cloud_score:.1f}%")
        print(f"  Resolution: {metadata.resolution}")
        
        cv2.imwrite("sample_output.png", image)
        print(f"\nSaved to: sample_output.png")
        
    except Exception as e:
        logger.error(f"Failed to fetch imagery: {e}")