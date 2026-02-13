import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from PIL import Image
import os
import tempfile

@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing"""
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(img_array)

@pytest.fixture
def sample_image_array():
    """Create a sample numpy array image"""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image"""
    img_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    return img_array

@pytest.fixture
def temp_image_path(sample_image):
    """Create a temporary image file"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        sample_image.save(f.name)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def mock_location():
    """Sample location coordinates"""
    return (34.0522, -118.2437)  # Los Angeles

@pytest.fixture
def mock_dates():
    """Sample before/after dates"""
    return {
        "before_date": "2024-01-01",
        "after_date": "2024-12-01"
    }

@pytest.fixture
def mock_analysis_input(mock_location, mock_dates):
    """Complete analysis input data"""
    return {
        "location": mock_location,
        "before_date": mock_dates["before_date"],
        "after_date": mock_dates["after_date"]
    }

@pytest.fixture
def mock_classification_result():
    """Sample classification result"""
    return {
        "label": "wildfire",
        "confidence": 0.89,
        "probabilities": {
            "wildfire": 0.89,
            "normal": 0.05,
            "urban": 0.03,
            "flood": 0.02,
            "drought": 0.01
        }
    }

@pytest.fixture
def mock_analysis_result(mock_location, mock_dates, mock_classification_result):
    """Complete analysis result"""
    return {
        "classification": mock_classification_result,
        "summary": "Significant wildfire activity detected in the region.",
        "solutions": "Immediate evacuation protocols recommended. Deploy fire suppression resources.",
        "images": {
            "before": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "after": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            "difference": np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        },
        "input_params": {
            "location": mock_location,
            "before_date": mock_dates["before_date"],
            "after_date": mock_dates["after_date"]
        },
        "timestamp": "2024-01-15T10:30:00"
    }

@pytest.fixture
def temp_knowledge_base():
    """Create temporary knowledge base file"""
    content = """
    WILDFIRE RESPONSE PROTOCOLS:
    - Immediate evacuation of affected areas
    - Deploy aerial fire suppression
    - Establish firebreaks
    
    FLOOD RESPONSE:
    - Emergency shelter setup
    - Sandbag deployment
    - Water rescue teams
    
    EARTHQUAKE RESPONSE:
    - Search and rescue operations
    - Structural assessments
    - Emergency medical services
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def test_cache_dir():
    """Create temporary cache directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
