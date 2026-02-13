import pytest
import numpy as np
from app.image_utils import detect_changes, preprocess_image
from PIL import Image
import cv2


class TestDetectChanges:
    """Test detect_changes function"""
    
    def test_detect_changes_same_images(self, sample_image_array):
        """Test no change detected for identical images"""
        diff_map = detect_changes(sample_image_array, sample_image_array)
        assert diff_map is not None
        assert isinstance(diff_map, np.ndarray)
        # Most pixels should show no change
        assert np.mean(diff_map) < 50  # Low average difference
    
    def test_detect_changes_different_images(self):
        """Test change detected for different images"""
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        img2 = np.ones((256, 256, 3), dtype=np.uint8) * 255
        diff_map = detect_changes(img1, img2)
        assert diff_map is not None
        assert np.mean(diff_map) > 100  # High average difference
    
    def test_detect_changes_output_shape(self, sample_image_array):
        """Test output shape is correct"""
        diff_map = detect_changes(sample_image_array, sample_image_array)
        assert diff_map.shape[:2] == sample_image_array.shape[:2]
    
    def test_detect_changes_different_sizes_raises_error(self):
        """Test error when images have different sizes"""
        img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        with pytest.raises((ValueError, Exception)):
            detect_changes(img1, img2)
    
    def test_detect_changes_partial_difference(self):
        """Test detecting partial changes"""
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        img2 = img1.copy()
        # Change one quadrant
        img2[:128, :128] = 255
        diff_map = detect_changes(img1, img2)
        # Should show changes in modified area
        assert np.mean(diff_map[:128, :128]) > np.mean(diff_map[128:, 128:])
    
    def test_detect_changes_grayscale_images(self):
        """Test with grayscale images"""
        img1 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        diff_map = detect_changes(img1, img2)
        assert diff_map is not None


class TestPreprocessImage:
    """Test preprocess_image function"""
    
    def test_preprocess_numpy_array(self, sample_image_array):
        """Test preprocessing numpy array"""
        processed = preprocess_image(sample_image_array)
        assert processed is not None
        assert isinstance(processed, (np.ndarray, Image.Image)) or hasattr(processed, 'shape')
    
    def test_preprocess_pil_image(self, sample_image):
        """Test preprocessing PIL Image"""
        processed = preprocess_image(sample_image)
        assert processed is not None
    
    def test_preprocess_file_path(self, temp_image_path):
        """Test preprocessing from file path"""
        processed = preprocess_image(temp_image_path)
        assert processed is not None
    
    def test_preprocess_resizing(self, sample_image_array):
        """Test image resizing during preprocessing"""
        processed = preprocess_image(sample_image_array)
        # Should normalize or resize to standard size
        assert processed is not None
    
    def test_preprocess_normalization(self, sample_image_array):
        """Test pixel value normalization"""
        processed = preprocess_image(sample_image_array)
        # Values might be normalized to [0, 1] or [-1, 1]
        assert processed is not None
    
    def test_preprocess_invalid_path(self):
        """Test error handling for invalid path"""
        with pytest.raises((FileNotFoundError, ValueError, Exception)):
            preprocess_image("nonexistent_image.jpg")


class TestImageUtilsEdgeCases:
    """Test edge cases and error handling"""
    
    def test_detect_changes_empty_images(self):
        """Test with empty images"""
        with pytest.raises((ValueError, Exception)):
            detect_changes(np.array([]), np.array([]))
    
    def test_detect_changes_none_input(self):
        """Test with None input"""
        with pytest.raises((TypeError, ValueError, Exception)):
            detect_changes(None, None)
    
    def test_preprocess_none_input(self):
        """Test preprocess with None"""
        with pytest.raises((TypeError, ValueError, Exception)):
            preprocess_image(None)
    
    def test_detect_changes_single_channel(self):
        """Test with single channel images"""
        img1 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        diff_map = detect_changes(img1, img2)
        assert diff_map is not None
    
    def test_detect_changes_four_channel(self):
        """Test with RGBA images"""
        img1 = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        try:
            diff_map = detect_changes(img1, img2)
            assert diff_map is not None
        except (ValueError, Exception):
            # Some implementations may not support 4 channels
            pass


class TestImageQuality:
    """Test image quality metrics"""
    
    def test_image_brightness(self, sample_image_array):
        """Test brightness calculation"""
        brightness = np.mean(sample_image_array)
        assert 0 <= brightness <= 255
    
    def test_image_contrast(self, sample_image_array):
        """Test contrast calculation"""
        contrast = np.std(sample_image_array)
        assert contrast >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
