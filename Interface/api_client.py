import requests
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class OrbitalWitnessAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_prefix = "/api/v1"
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{self.api_prefix}{endpoint}"
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise Exception(f"API Error: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        return self._make_request("GET", "/health")
    
    def detailed_health_check(self) -> Dict[str, Any]:
        return self._make_request("GET", "/health/detailed")
    
    def analyze(
        self,
        location: Tuple[float, float],
        before_date: str,
        after_date: str
    ) -> Dict[str, Any]:
        payload = {
            "location": location,
            "before_date": before_date,
            "after_date": after_date
        }
        return self._make_request("POST", "/analyze", json=payload, timeout=300)
    
    def analyze_async(
        self,
        location: Tuple[float, float],
        before_date: str,
        after_date: str
    ) -> Dict[str, Any]:
        payload = {
            "location": location,
            "before_date": before_date,
            "after_date": after_date
        }
        return self._make_request("POST", "/analyze/async", json=payload)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        return self._make_request("GET", f"/analyze/task/{task_id}")
    
    def classify(
        self,
        location: Tuple[float, float],
        before_date: str,
        after_date: str
    ) -> Dict[str, Any]:
        payload = {
            "location": location,
            "before_date": before_date,
            "after_date": after_date
        }
        return self._make_request("POST", "/classify", json=payload, timeout=300)
    
    def is_api_available(self) -> bool:
        try:
            self.health_check()
            return True
        except:
            return False
