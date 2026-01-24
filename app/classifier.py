import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DisasterType(Enum):
    DEFORESTATION = "deforestation"
    URBAN_DEVELOPMENT = "urban"
    FLOOD = "flood"
    NORMAL = "normal"
    WILDFIRE = "wildfire"
    VOLCANIC_ERUPTION = "volcanic_eruption"
    BOMBARDMENT = "bombardment"
    EARTHQUAKE = "earthquake"
    LANDSLIDE = "landslide"
    DROUGHT = "drought"


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    top_k_predictions: List[Tuple[str, float]]
    features: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "top_k_predictions": [
                {"label": label, "confidence": round(conf, 4)} 
                for label, conf in self.top_k_predictions
            ],
            "metadata": self.metadata or {}
        }


class ModelBackbone(Enum):
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B4 = "efficientnet_b4"
    VGG16 = "vgg16"
    DENSENET121 = "densenet121"
    MOBILENET_V2 = "mobilenet_v2"


class SatelliteImageClassifier:
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        backbone: ModelBackbone = ModelBackbone.RESNET50,
        num_classes: int = 10,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        use_pretrained: bool = True
    ):
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = backbone
        
        self.class_labels = [e.value for e in DisasterType][:num_classes]
        
        logger.info(f"Initializing {backbone.value} on {self.device}")
        self.model = self._build_model(backbone, use_pretrained)
        self.model = self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            self._load_weights(model_path)
        else:
            logger.warning("No pretrained weights loaded. Using random initialization.")
        
        self.model.eval()
        
        self.transform = self._get_transform()
    
    def _build_model(self, backbone: ModelBackbone, use_pretrained: bool) -> nn.Module:
        if backbone == ModelBackbone.RESNET50:
            model = models.resnet50(pretrained=use_pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            
        elif backbone == ModelBackbone.RESNET101:
            model = models.resnet101(pretrained=use_pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            
        elif backbone == ModelBackbone.EFFICIENTNET_B0:
            model = models.efficientnet_b0(pretrained=use_pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            
        elif backbone == ModelBackbone.EFFICIENTNET_B4:
            model = models.efficientnet_b4(pretrained=use_pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            
        elif backbone == ModelBackbone.VGG16:
            model = models.vgg16(pretrained=use_pretrained)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
            
        elif backbone == ModelBackbone.DENSENET121:
            model = models.densenet121(pretrained=use_pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
            
        elif backbone == ModelBackbone.MOBILENET_V2:
            model = models.mobilenet_v2(pretrained=use_pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return model
    
    def _get_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_weights(self, model_path: str):
        try:
            logger.info(f"Loading weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            logger.info("Weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load weights: {str(e)}")
            raise
    
    def preprocess_image(
        self, 
        image: Union[np.ndarray, str, Image.Image]
    ) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        tensor = self.transform(image)
        return tensor.unsqueeze(0)
    
    @torch.no_grad()
    def classify_image(
        self,
        image: Union[np.ndarray, str, Image.Image],
        top_k: int = 3,
        return_features: bool = False
    ) -> ClassificationResult:
        try:
            input_tensor = self.preprocess_image(image).to(self.device)
            
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
            
            top_k_predictions = [
                (self.class_labels[idx], float(prob))
                for idx, prob in zip(top_indices, top_probs)
            ]
            
            primary_label = top_k_predictions[0][0]
            primary_confidence = top_k_predictions[0][1]
            
            features = None
            if return_features:
                features = self._extract_features(input_tensor)
            
            metadata = {
                "model_backbone": self.backbone.value,
                "device": self.device,
                "image_size": input_tensor.shape[-2:],
                "confidence_threshold": self.confidence_threshold,
                "meets_threshold": primary_confidence >= self.confidence_threshold
            }
            
            return ClassificationResult(
                label=primary_label,
                confidence=primary_confidence,
                top_k_predictions=top_k_predictions,
                features=features,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            raise
    
    def _extract_features(self, input_tensor: torch.Tensor) -> np.ndarray:
        if hasattr(self.model, 'fc'):
            feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        elif hasattr(self.model, 'classifier'):
            feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        else:
            return None
        
        features = feature_extractor(input_tensor)
        features = features.squeeze().cpu().numpy()
        return features
    
    def classify_batch(
        self,
        images: List[Union[np.ndarray, str, Image.Image]],
        batch_size: int = 32,
        top_k: int = 3
    ) -> List[ClassificationResult]:
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = [
                self.classify_image(img, top_k=top_k) 
                for img in batch
            ]
            results.extend(batch_results)
        
        return results
    
    def save_model(self, save_path: str):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_labels': self.class_labels,
            'backbone': self.backbone.value,
            'num_classes': self.num_classes
        }, save_path)
        logger.info(f"Model saved to {save_path}")


class EnsembleClassifier:
    
    def __init__(self, classifiers: List[SatelliteImageClassifier]):
        self.classifiers = classifiers
        logger.info(f"Initialized ensemble with {len(classifiers)} models")
    
    def classify_image(
        self, 
        image: Union[np.ndarray, str, Image.Image],
        voting: str = 'soft'
    ) -> ClassificationResult:
        
        predictions = [clf.classify_image(image) for clf in self.classifiers]
        
        if voting == 'soft':
            # Average probabilities
            all_labels = predictions[0].top_k_predictions
            label_scores = {}
            
            for pred in predictions:
                for label, conf in pred.top_k_predictions:
                    label_scores[label] = label_scores.get(label, 0) + conf
            
            # Average and sort
            for label in label_scores:
                label_scores[label] /= len(self.classifiers)
            
            sorted_predictions = sorted(
                label_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return ClassificationResult(
                label=sorted_predictions[0][0],
                confidence=sorted_predictions[0][1],
                top_k_predictions=sorted_predictions[:3],
                metadata={"ensemble_method": "soft_voting"}
            )
        
        else:
            votes = {}
            for pred in predictions:
                label = pred.label
                votes[label] = votes.get(label, 0) + 1
            
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            winning_label = sorted_votes[0][0]
            
            avg_confidence = np.mean([
                pred.confidence for pred in predictions 
                if pred.label == winning_label
            ])
            
            return ClassificationResult(
                label=winning_label,
                confidence=avg_confidence,
                top_k_predictions=sorted_votes[:3],
                metadata={"ensemble_method": "hard_voting"}
            )


def classify_image(
    image_np: np.ndarray,
    model_path: Optional[str] = None,
    backbone: str = "resnet50"
) -> Dict:
    try:
        backbone_enum = ModelBackbone(backbone)
    except ValueError:
        backbone_enum = ModelBackbone.RESNET50
    
    classifier = SatelliteImageClassifier(
        model_path=model_path,
        backbone=backbone_enum
    )
    
    result = classifier.classify_image(image_np)
    return result.to_dict()


if __name__ == "__main__":
    classifier = SatelliteImageClassifier(
        backbone=ModelBackbone.RESNET50,
        num_classes=10,
        confidence_threshold=0.6
    )
    
    dummy_image = np.random.rand(256, 256, 3)
    result = classifier.classify_image(dummy_image, top_k=3)
    
    print("Classification Results:")
    print(f"Primary Label: {result.label}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nTop-3 Predictions:")
    for label, conf in result.top_k_predictions:
        print(f"  {label}: {conf:.2%}")
    
    classifiers = [
        SatelliteImageClassifier(backbone=ModelBackbone.RESNET50),
        SatelliteImageClassifier(backbone=ModelBackbone.EFFICIENTNET_B0),
        SatelliteImageClassifier(backbone=ModelBackbone.DENSENET121)
    ]
    ensemble = EnsembleClassifier(classifiers)
    ensemble_result = ensemble.classify_image(dummy_image, voting='soft')
    
    print(f"\nEnsemble Result: {ensemble_result.label} ({ensemble_result.confidence:.2%})")