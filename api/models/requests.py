from pydantic import BaseModel, Field, validator
from typing import Tuple, Optional

class AnalysisRequest(BaseModel):
    location: Tuple[float, float] = Field(
        ...,
        description="Latitude and longitude coordinates",
        example=(40.7128, -74.0060)
    )
    before_date: str = Field(
        ...,
        description="Baseline date for comparison (YYYY-MM-DD)",
        example="2024-01-01"
    )
    after_date: str = Field(
        ...,
        description="Analysis date (YYYY-MM-DD)",
        example="2024-12-01"
    )
    
    @validator('location')
    def validate_location(cls, v):
        lat, lon = v
        if not (-90 <= lat <= 90):
            raise ValueError('Latitude must be between -90 and 90')
        if not (-180 <= lon <= 180):
            raise ValueError('Longitude must be between -180 and 180')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": (34.0522, -118.2437),
                "before_date": "2024-01-01",
                "after_date": "2024-12-01"
            }
        }

class ClassificationRequest(BaseModel):
    location: Tuple[float, float] = Field(
        ...,
        description="Latitude and longitude coordinates"
    )
    before_date: str = Field(
        ...,
        description="Baseline date (YYYY-MM-DD)"
    )
    after_date: str = Field(
        ...,
        description="Analysis date (YYYY-MM-DD)"
    )
    
    @validator('location')
    def validate_location(cls, v):
        lat, lon = v
        if not (-90 <= lat <= 90):
            raise ValueError('Latitude must be between -90 and 90')
        if not (-180 <= lon <= 180):
            raise ValueError('Longitude must be between -180 and 180')
        return v