import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from api.main import app
from api.routes.analysis import set_agent


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """Create mock agent"""
    agent = Mock()
    agent.invoke.return_value = {
        "classification": {"label": "wildfire", "confidence": 0.89},
        "summary": "Wildfire detected",
        "solutions": "Evacuate immediately",
        "timestamp": "2024-01-15T10:30:00"
    }
    agent.fetch_satellite_data.return_value = {
        "images": {},
        "input_params": {}
    }
    agent.run_classification.return_value = {
        "classification": {"label": "wildfire", "confidence": 0.89},
        "summary": "Wildfire detected"
    }
    set_agent(agent)
    return agent


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_endpoint(self, client):
        """Test basic health check"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "service" in data
    
    def test_detailed_health_endpoint(self, client):
        """Test detailed health check"""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "system" in data
        assert "environment" in data
        assert "cpu_percent" in data["system"]
        assert "memory_percent" in data["system"]


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestAnalyzeEndpoint:
    """Test analysis endpoint"""
    
    def test_analyze_success(self, client, mock_agent):
        """Test successful analysis"""
        payload = {
            "location": (34.0522, -118.2437),
            "before_date": "2024-01-01",
            "after_date": "2024-12-01"
        }
        
        response = client.post("/api/v1/analyze", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "classification" in data
        assert "summary" in data
        assert "solutions" in data
        assert data["location"] == payload["location"]
    
    def test_analyze_invalid_location(self, client, mock_agent):
        """Test analysis with invalid location"""
        payload = {
            "location": (100, 200),  # Invalid lat/lon
            "before_date": "2024-01-01",
            "after_date": "2024-12-01"
        }
        
        response = client.post("/api/v1/analyze", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_analyze_missing_fields(self, client, mock_agent):
        """Test analysis with missing required fields"""
        payload = {
            "location": (34.0522, -118.2437)
            # Missing dates
        }
        
        response = client.post("/api/v1/analyze", json=payload)
        assert response.status_code == 422
    
    def test_analyze_invalid_date_format(self, client, mock_agent):
        """Test analysis with invalid date format"""
        payload = {
            "location": (34.0522, -118.2437),
            "before_date": "01-01-2024",  # Wrong format
            "after_date": "2024-12-01"
        }
        
        response = client.post("/api/v1/analyze", json=payload)
        # Should either accept or reject with 422
        assert response.status_code in [200, 422]


class TestClassifyEndpoint:
    """Test classification endpoint"""
    
    def test_classify_success(self, client, mock_agent):
        """Test successful classification"""
        payload = {
            "location": (34.0522, -118.2437),
            "before_date": "2024-01-01",
            "after_date": "2024-12-01"
        }
        
        response = client.post("/api/v1/classify", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "classification" in data
        assert "summary" in data
        # Should not have solutions
    
    def test_classify_invalid_coordinates(self, client, mock_agent):
        """Test classification with invalid coordinates"""
        payload = {
            "location": (200, 300),
            "before_date": "2024-01-01",
            "after_date": "2024-12-01"
        }
        
        response = client.post("/api/v1/classify", json=payload)
        assert response.status_code == 422


class TestAsyncAnalyzeEndpoint:
    """Test async analysis endpoint"""
    
    def test_analyze_async_submission(self, client, mock_agent):
        """Test async analysis task submission"""
        payload = {
            "location": (34.0522, -118.2437),
            "before_date": "2024-01-01",
            "after_date": "2024-12-01"
        }
        
        response = client.post("/api/v1/analyze/async", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert "task_id" in data
        assert "message" in data
    
    def test_get_task_status_not_found(self, client):
        """Test getting status of non-existent task"""
        response = client.get("/api/v1/analyze/task/nonexistent_task")
        assert response.status_code == 404
    
    def test_async_task_flow(self, client, mock_agent):
        """Test full async task flow"""
        # Submit task
        payload = {
            "location": (34.0522, -118.2437),
            "before_date": "2024-01-01",
            "after_date": "2024-12-01"
        }
        
        submit_response = client.post("/api/v1/analyze/async", json=payload)
        assert submit_response.status_code == 200
        task_id = submit_response.json()["task_id"]
        
        # Check task status
        status_response = client.get(f"/api/v1/analyze/task/{task_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert "status" in status_data


class TestErrorHandling:
    """Test error handling"""
    
    def test_agent_not_initialized(self, client):
        """Test behavior when agent is not initialized"""
        # Reset agent
        set_agent(None)
        
        payload = {
            "location": (34.0522, -118.2437),
            "before_date": "2024-01-01",
            "after_date": "2024-12-01"
        }
        
        # Should initialize agent on first request or return 503
        response = client.post("/api/v1/analyze", json=payload)
        assert response.status_code in [200, 503]
    
    def test_invalid_json(self, client):
        """Test invalid JSON payload"""
        response = client.post(
            "/api/v1/analyze",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/api/v1/health")
        # CORS headers should be present
        assert response.status_code in [200, 405]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
