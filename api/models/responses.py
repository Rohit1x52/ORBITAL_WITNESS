from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Tuple

class AnalysisResponse(BaseModel):
    status: str = Field(..., description="Response status")
    timestamp: str = Field(..., description="Response timestamp")
    location: Tuple[float, float] = Field(..., description="Analysis location")
    analysis_period: Dict[str, str] = Field(..., description="Analysis time period")
    classification: Dict[str, Any] = Field(..., description="Classification results")
    summary: str = Field(..., description="Analysis summary")
    solutions: str = Field(..., description="Recommended solutions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2024-01-15T10:30:00",
                "location": (34.0522, -118.2437),
                "analysis_period": {
                    "before": "2024-01-01",
                    "after": "2024-12-01"
                },
                "classification": {
                    "label": "wildfire",
                    "confidence": 0.89
                },
                "summary": "Significant wildfire activity detected in the region.",
                "solutions": "Immediate evacuation protocols recommended...",
                "metadata": {
                    "processing_time": "2024-01-15T10:29:45",
                    "confidence": 0.89
                }
            }
        }

class ClassificationResponse(BaseModel):
    status: str
    timestamp: str
    location: Tuple[float, float]
    classification: Dict[str, Any]
    summary: str

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    graph_run_id: Optional[str] = None
    
class ErrorResponse(BaseModel):
    detail: str
    error: Optional[str] = None
    timestamp: str