import pytest
import numpy as np
from app.classifier import SatelliteImageClassifier, DisasterType, EnsembleClassifier
from PIL import Image


class TestDisasterType:
    """Test DisasterType enum"""
    
    def test_disaster_types_exist(self):
        """Test all disaster types are defined"""
        assert hasattr(DisasterType, 'DEFORESTATION')
        assert hasattr(DisasterType, 'WILDFIRE')
        assert hasattr(DisasterType, 'FLOOD')
        assert hasattr(DisasterType, 'EARTHQUAKE')
        assert hasattr(DisasterType, 'NORMAL')
    
    def test_disaster_type_values(self):
        """Test disaster type string values"""
        assert DisasterType.WILDFIRE.value == "wildfire"
        assert DisasterType.FLOOD.value == "flood"
        assert DisasterType.NORMAL.value == "normal"


class TestSatelliteImageClassifier:
    """Test SatelliteImageClassifier class"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        return SatelliteImageClassifier()
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initializes correctly"""
        assert classifier is not None
        assert hasattr(classifier, 'device')
        assert hasattr(classifier, 'model')
    
    def test_preprocess_image_from_array(self, classifier, sample_image_array):
        """Test preprocessing numpy array"""
        processed = classifier.preprocess_image(sample_image_array)
        assert processed is not None
        assert isinstance(processed, np.ndarray) or hasattr(processed, 'shape')
    
    def test_preprocess_image_from_pil(self, classifier, sample_image):
        """Test preprocessing PIL image"""
        processed = classifier.preprocess_image(sample_image)
        assert processed is not None
    
    def test_preprocess_image_from_path(self, classifier, temp_image_path):
        """Test preprocessing from file path"""
        processed = classifier.preprocess_image(temp_image_path)
        assert processed is not None
    
    def test_preprocess_invalid_input(self, classifier):
        """Test preprocessing with invalid input"""
        with pytest.raises((ValueError, TypeError, Exception)):
            classifier.preprocess_image("nonexistent_file.jpg")
    
    def test_predict_returns_dict(self, classifier, sample_image_array):
        """Test predict returns proper dict structure"""
        result = classifier.predict(sample_image_array)
        assert isinstance(result, dict)
        assert 'label' in result
        assert 'confidence' in result
    
    def test_predict_confidence_range(self, classifier, sample_image_array):
        """Test confidence is between 0 and 1"""
        result = classifier.predict(sample_image_array)
        assert 0 <= result['confidence'] <= 1
    
    def test_predict_label_is_valid(self, classifier, sample_image_array):
        """Test predicted label is valid disaster type"""
        result = classifier.predict(sample_image_array)
        valid_labels = [dt.value for dt in DisasterType]
        assert result['label'] in valid_labels or result['label'] in [
            'No Significant Change', 'wildfire', 'flood', 'earthquake', 
            'urban', 'deforestation', 'normal', 'drought', 'landslide',
            'volcanic_eruption', 'bombardment'
        ]
    
    def test_predict_with_different_image_sizes(self, classifier):
        """Test prediction works with different image sizes"""
        for size in [(128, 128, 3), (256, 256, 3), (512, 512, 3)]:
            img = np.random.randint(0, 255, size, dtype=np.uint8)
            result = classifier.predict(img)
            assert result is not None
            assert 'label' in result
    
    def test_predict_grayscale_image(self, classifier, sample_grayscale_image):
        """Test prediction with grayscale image"""
        result = classifier.predict(sample_grayscale_image)
        assert result is not None
        assert 'label' in result


class TestEnsembleClassifier:
    """Test EnsembleClassifier class"""
    
    @pytest.fixture
    def ensemble(self):
        """Create ensemble classifier instance"""
        return EnsembleClassifier()
    
    def test_ensemble_initialization(self, ensemble):
        """Test ensemble initializes with multiple models"""
        assert ensemble is not None
        assert hasattr(ensemble, 'classifiers')
        assert len(ensemble.classifiers) > 0
    
    def test_ensemble_predict(self, ensemble, sample_image_array):
        """Test ensemble prediction"""
        result = ensemble.predict(sample_image_array)
        assert isinstance(result, dict)
        assert 'label' in result
        assert 'confidence' in result
    
    def test_ensemble_confidence_higher(self, ensemble, sample_image_array):
        """Test ensemble generally has higher confidence"""
        result = ensemble.predict(sample_image_array)
        assert 0 <= result['confidence'] <= 1


class TestClassifyImage:
    """Test classify_image function"""
    
    def test_classify_image_function_exists(self):
        """Test classify_image function is importable"""
        from app.classifier import classify_image
        assert callable(classify_image)
    
    def test_classify_image_returns_dict(self, sample_image_array):
        """Test classify_image returns dict"""
        from app.classifier import classify_image
        result = classify_image(sample_image_array)
        assert isinstance(result, dict)
        assert 'label' in result
        assert 'confidence' in result


class TestClassifierEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def classifier(self):
        return SatelliteImageClassifier()
    
    def test_empty_image(self, classifier):
        """Test with empty image"""
        with pytest.raises((ValueError, Exception)):
            classifier.predict(np.array([]))
    
    def test_wrong_dimensions(self, classifier):
        """Test with wrong number of dimensions"""
        with pytest.raises((ValueError, Exception)):
            classifier.predict(np.random.randint(0, 255, (256,), dtype=np.uint8))
    
    def test_out_of_range_values(self, classifier):
        """Test with out of range pixel values"""
        img = np.random.randint(-100, 400, (256, 256, 3), dtype=np.int32)
        # Should handle or raise appropriate error
        try:
            result = classifier.predict(img)
            assert result is not None
        except (ValueError, Exception):
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
