"""
Advanced image processing utilities for satellite imagery analysis
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChangeDetectionMethod(Enum):
    SIMPLE_DIFF = "simple_diff"
    STRUCTURAL_SIMILARITY = "ssim"
    IMAGE_RATIO = "image_ratio"
    HISTOGRAM_COMPARISON = "histogram"
    OPTICAL_FLOW = "optical_flow"
    MORPHOLOGICAL = "morphological"


class InterpolationMethod(Enum):
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4


@dataclass
class ChangeDetectionConfig:
    method: ChangeDetectionMethod = ChangeDetectionMethod.SIMPLE_DIFF
    threshold: int = 30
    gaussian_blur: bool = True
    blur_kernel: Tuple[int, int] = (5, 5)
    morphological_ops: bool = True
    morph_kernel_size: int = 3
    min_change_area: int = 100
    output_size: Tuple[int, int] = (256, 256)
    interpolation: InterpolationMethod = InterpolationMethod.AREA


@dataclass
class PreprocessConfig:
    target_size: Tuple[int, int] = (224, 224)
    interpolation: InterpolationMethod = InterpolationMethod.AREA
    normalize: bool = True
    normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    color_correction: bool = False
    denoise: bool = False
    enhance_contrast: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)


class ImageProcessor:
    def __init__(
        self,
        change_config: Optional[ChangeDetectionConfig] = None,
        preprocess_config: Optional[PreprocessConfig] = None
    ):
        self.change_config = change_config or ChangeDetectionConfig()
        self.preprocess_config = preprocess_config or PreprocessConfig()
    
    def validate_images(
        self, 
        before_image: np.ndarray, 
        after_image: np.ndarray
    ) -> bool:
        if before_image is None or after_image is None:
            raise ValueError("Input images cannot be None")
        
        if before_image.shape != after_image.shape:
            logger.warning(
                f"Image shape mismatch: {before_image.shape} vs {after_image.shape}. "
                "Resizing to match."
            )
            return False
        
        if len(before_image.shape) not in [2, 3]:
            raise ValueError(f"Invalid image dimensions: {before_image.shape}")
        
        return True
    
    def align_images(
        self, 
        before_image: np.ndarray, 
        after_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_shape = before_image.shape
        
        if after_image.shape != target_shape:
            after_image = cv2.resize(
                after_image,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_AREA
            )
        
        return before_image, after_image
    
    def _apply_gaussian_blur(
        self, 
        image: np.ndarray, 
        kernel: Tuple[int, int]
    ) -> np.ndarray:
        return cv2.GaussianBlur(image, kernel, 0)
    
    def _apply_morphological_ops(
        self, 
        image: np.ndarray, 
        kernel_size: int
    ) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        return image
    
    def _remove_small_regions(
        self, 
        binary_image: np.ndarray, 
        min_area: int
    ) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        
        filtered_image = np.zeros_like(binary_image)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_image[labels == i] = 255
        
        return filtered_image
    
    def detect_changes_simple_diff(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray
    ) -> np.ndarray:
        if self.change_config.gaussian_blur:
            before_image = self._apply_gaussian_blur(
                before_image, 
                self.change_config.blur_kernel
            )
            after_image = self._apply_gaussian_blur(
                after_image, 
                self.change_config.blur_kernel
            )
        
        diff = cv2.absdiff(before_image, after_image)
        
        if len(diff.shape) == 3:
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            gray_diff = diff
        
        _, threshold_diff = cv2.threshold(
            gray_diff,
            self.change_config.threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        if self.change_config.morphological_ops:
            threshold_diff = self._apply_morphological_ops(
                threshold_diff,
                self.change_config.morph_kernel_size
            )
        
        if self.change_config.min_change_area > 0:
            threshold_diff = self._remove_small_regions(
                threshold_diff,
                self.change_config.min_change_area
            )
        
        return threshold_diff
    
    def detect_changes_ssim(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray
    ) -> np.ndarray:
        from skimage.metrics import structural_similarity
        
        if len(before_image.shape) == 3:
            before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)
        else:
            before_gray = before_image
            after_gray = after_image
        
        score, diff = structural_similarity(
            before_gray,
            after_gray,
            full=True
        )
        
        diff = (1 - diff) * 255
        diff = diff.astype(np.uint8)
        
        _, threshold_diff = cv2.threshold(
            diff,
            self.change_config.threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        return threshold_diff
    
    def detect_changes_image_ratio(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray
    ) -> np.ndarray:
        before_float = before_image.astype(np.float32) + 1e-6
        after_float = after_image.astype(np.float32) + 1e-6
        
        ratio = np.abs(np.log(after_float / before_float))
        
        if len(ratio.shape) == 3:
            ratio_gray = cv2.cvtColor(
                ratio.astype(np.uint8), 
                cv2.COLOR_BGR2GRAY
            )
        else:
            ratio_gray = ratio
        
        ratio_normalized = cv2.normalize(
            ratio_gray, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        
        _, threshold_diff = cv2.threshold(
            ratio_normalized,
            self.change_config.threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        return threshold_diff
    
    def detect_changes_histogram(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray
    ) -> np.ndarray:
        if len(before_image.shape) == 3:
            before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)
        else:
            before_gray = before_image
            after_gray = after_image
        
        hist_before = cv2.calcHist([before_gray], [0], None, [256], [0, 256])
        hist_after = cv2.calcHist([after_gray], [0], None, [256], [0, 256])
        
        cv2.normalize(hist_before, hist_before, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_after, hist_after, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        correlation = cv2.compareHist(hist_before, hist_after, cv2.HISTCMP_CORREL)
        
        diff = cv2.absdiff(before_gray, after_gray)
        _, threshold_diff = cv2.threshold(
            diff,
            int(self.change_config.threshold * (1 - correlation)),
            255,
            cv2.THRESH_BINARY
        )
        
        return threshold_diff
    
    def detect_changes(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        return_metrics: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        logger.info(f"Detecting changes using {self.change_config.method.value} method")
        
        try:
            self.validate_images(before_image, after_image)
        except ValueError:
            pass
        
        before_image, after_image = self.align_images(before_image, after_image)
        
        if self.change_config.method == ChangeDetectionMethod.SIMPLE_DIFF:
            diff_map = self.detect_changes_simple_diff(before_image, after_image)
        elif self.change_config.method == ChangeDetectionMethod.STRUCTURAL_SIMILARITY:
            diff_map = self.detect_changes_ssim(before_image, after_image)
        elif self.change_config.method == ChangeDetectionMethod.IMAGE_RATIO:
            diff_map = self.detect_changes_image_ratio(before_image, after_image)
        elif self.change_config.method == ChangeDetectionMethod.HISTOGRAM_COMPARISON:
            diff_map = self.detect_changes_histogram(before_image, after_image)
        else:
            diff_map = self.detect_changes_simple_diff(before_image, after_image)
        
        resized_diff = cv2.resize(
            diff_map,
            self.change_config.output_size,
            interpolation=self.change_config.interpolation.value
        )
        
        if len(resized_diff.shape) == 2:
            final_diff_map = cv2.cvtColor(resized_diff, cv2.COLOR_GRAY2BGR)
        else:
            final_diff_map = resized_diff
        
        if return_metrics:
            metrics = self._calculate_change_metrics(diff_map)
            logger.info(f"Change detection complete. Changed pixels: {metrics['changed_pixels']}")
            return final_diff_map, metrics
        
        logger.info("Change detection complete")
        return final_diff_map
    
    def _calculate_change_metrics(self, diff_map: np.ndarray) -> Dict:
        total_pixels = diff_map.shape[0] * diff_map.shape[1]
        changed_pixels = np.count_nonzero(diff_map)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        return {
            'changed_pixels': int(changed_pixels),
            'total_pixels': int(total_pixels),
            'change_percentage': float(change_percentage),
            'unchanged_pixels': int(total_pixels - changed_pixels)
        }
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self.preprocess_config.clahe_clip_limit,
            tileGridSize=self.preprocess_config.clahe_tile_size
        )
        
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            image = clahe.apply(image)
        
        return image
    
    def _apply_color_correction(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) != 3:
            return image
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        l = cv2.equalizeHist(l)
        
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return image
    
    def _apply_denoising(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return image
    
    def preprocess_image(
        self,
        image: np.ndarray,
        return_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        logger.info(f"Preprocessing image to size {self.preprocess_config.target_size}")
        
        if image is None:
            raise ValueError("Input image cannot be None")
        
        original_shape = image.shape
        
        if self.preprocess_config.denoise:
            image = self._apply_denoising(image)
        
        if self.preprocess_config.color_correction:
            image = self._apply_color_correction(image)
        
        if self.preprocess_config.enhance_contrast:
            image = self._apply_clahe(image)
        
        resized_image = cv2.resize(
            image,
            self.preprocess_config.target_size,
            interpolation=self.preprocess_config.interpolation.value
        )
        
        if self.preprocess_config.normalize:
            resized_image = resized_image.astype(np.float32) / 255.0
            
            if len(resized_image.shape) == 3:
                mean = np.array(self.preprocess_config.normalization_mean)
                std = np.array(self.preprocess_config.normalization_std)
                resized_image = (resized_image - mean) / std
        
        if return_metadata:
            metadata = {
                'original_shape': original_shape,
                'preprocessed_shape': resized_image.shape,
                'normalized': self.preprocess_config.normalize,
                'denoised': self.preprocess_config.denoise,
                'color_corrected': self.preprocess_config.color_correction,
                'contrast_enhanced': self.preprocess_config.enhance_contrast
            }
            logger.info("Preprocessing complete")
            return resized_image, metadata
        
        logger.info("Preprocessing complete")
        return resized_image


def detect_changes(
    before_image: np.ndarray,
    after_image: np.ndarray,
    method: str = "simple_diff",
    threshold: int = 30,
    output_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    try:
        method_enum = ChangeDetectionMethod(method)
    except ValueError:
        method_enum = ChangeDetectionMethod.SIMPLE_DIFF
    
    config = ChangeDetectionConfig(
        method=method_enum,
        threshold=threshold,
        output_size=output_size
    )
    
    processor = ImageProcessor(change_config=config)
    return processor.detect_changes(before_image, after_image)


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = False
) -> np.ndarray:
    config = PreprocessConfig(
        target_size=target_size,
        normalize=normalize
    )
    
    processor = ImageProcessor(preprocess_config=config)
    return processor.preprocess_image(image)


if __name__ == "__main__":
    processor = ImageProcessor()
    
    before = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    after = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    diff_map, metrics = processor.detect_changes(
        before, after, return_metrics=True
    )
    
    print(f"Change Detection Metrics: {metrics}")
    
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    preprocessed, metadata = processor.preprocess_image(
        test_image, return_metadata=True
    )
    
    print(f"Preprocessing Metadata: {metadata}")