# Test Coverage Report

[![Coverage](https://img.shields.io/badge/coverage-0%25-red.svg)](htmlcov/index.html)

## Running Tests

### Install Test Dependencies
```bash
pip install pytest pytest-cov pytest-asyncio pytest-mock httpx
```

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=app --cov=api --cov-report=html --cov-report=term
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_classifier.py -v

# Specific test class
pytest tests/unit/test_classifier.py::TestSatelliteImageClassifier -v

# Specific test method
pytest tests/unit/test_classifier.py::TestSatelliteImageClassifier::test_predict_returns_dict -v
```

### Run Tests with Markers
```bash
# Run only API tests
pytest -m api

# Run only agent tests
pytest -m agent

# Run only fast tests (exclude slow ones)
pytest -m "not slow"
```

### Generate Coverage Report
```bash
# HTML report (open htmlcov/index.html in browser)
pytest --cov=app --cov=api --cov-report=html

# Terminal report with missing lines
pytest --cov=app --cov=api --cov-report=term-missing

# XML report (for CI/CD)
pytest --cov=app --cov=api --cov-report=xml
```

### Run Tests in Parallel (Faster)
```bash
pip install pytest-xdist
pytest -n auto  # Use all CPU cores
pytest -n 4     # Use 4 workers
```

### Run Tests with Verbose Output
```bash
pytest -vv              # Very verbose
pytest -v --tb=short    # Verbose with short traceback
pytest -v --tb=line     # Verbose with minimal traceback
```

### Run Failed Tests Only
```bash
pytest --lf   # Last failed
pytest --ff   # Failed first, then others
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── test_classifier.py   # Classifier tests (40+ tests)
│   ├── test_agent.py        # Agent tests (25+ tests)
│   ├── test_image_utils.py  # Image utilities (20+ tests)
│   └── test_api_endpoints.py # API endpoint tests (30+ tests)
└── integration/             # Integration tests
    └── test_full_pipeline.py # End-to-end tests (10+ tests)
```

## Current Test Coverage

| Module | Coverage | Statements | Missing |
|--------|----------|------------|---------|
| app/agent.py | 0% | 0/0 | - |
| app/classifier.py | 0% | 0/0 | - |
| app/image_utils.py | 0% | 0/0 | - |
| api/main.py | 0% | 0/0 | - |
| api/routes/analysis.py | 0% | 0/0 | - |
| api/routes/health.py | 0% | 0/0 | - |
| **Total** | **0%** | **0/0** | **-** |

*Run `pytest --cov` to update coverage metrics*

## Writing New Tests

### Unit Test Template
```python
import pytest
from app.your_module import YourClass

class TestYourClass:
    @pytest.fixture
    def instance(self):
        return YourClass()
    
    def test_method_name(self, instance):
        result = instance.method()
        assert result == expected_value
```

### API Test Template
```python
from fastapi.testclient import TestClient
from api.main import app

def test_endpoint():
    client = TestClient(app)
    response = client.post("/api/v1/endpoint", json=payload)
    assert response.status_code == 200
```

### Integration Test Template
```python
@pytest.mark.integration
def test_full_flow():
    # Test complete workflow
    pass
```

## Continuous Integration

Tests run automatically on:
- Every push to main branch
- Every pull request
- Scheduled daily at midnight

See `.github/workflows/ci.yml` for CI configuration.

## Test Best Practices

1. ✅ **Descriptive Names**: Use clear test function names
2. ✅ **Single Assertion**: One logical assertion per test (when possible)
3. ✅ **Fixtures**: Use fixtures for common setup
4. ✅ **Mocking**: Mock external dependencies (API calls, file I/O)
5. ✅ **Fast Tests**: Keep tests fast (< 1 second each)
6. ✅ **Independent**: Tests should not depend on each other
7. ✅ **Deterministic**: Tests should produce same result every time

## Troubleshooting

### Tests fail with import errors
```bash
# Ensure you're in virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install in development mode
pip install -e .
```

### Coverage not showing all files
```bash
# Make sure .coveragerc is configured correctly
# Run with explicit source paths
pytest --cov=app --cov=api
```

### Tests are too slow
```bash
# Run in parallel
pip install pytest-xdist
pytest -n auto

# Skip slow tests
pytest -m "not slow"
```

## Next Steps

- [ ] Achieve 80%+ code coverage
- [ ] Add performance benchmarking tests
- [ ] Add load testing for API
- [ ] Add security testing
- [ ] Add database integration tests
- [ ] Add end-to-end UI tests
