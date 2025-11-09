"""
Pydantic schemas for request and response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional


class ResumeText(BaseModel):
    text: str = Field(..., min_length=10, description="Resume text to classify")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Software Engineer with 5 years of experience in Python, Django, and FastAPI. Developed REST APIs and microservices..."
            }
        }


class PredictionResponse(BaseModel):
    category: str = Field(..., description="Predicted resume category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence level")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for all categories")
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "Data Science",
                "confidence": 0.89,
                "probabilities": {
                    "Data Science": 0.89,
                    "Web Developing": 0.05,
                    "Python Developer": 0.03,
                    "Other": 0.03
                }
            }
        }


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    message: Optional[str] = None


class CategoriesResponse(BaseModel):
    categories: list[str]
    total: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "categories": ["Data Science", "Web Developing", "Python Developer"],
                "total": 3
            }
        }


class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Error al procesar el archivo",
                "error_type": "ProcessingError"
            }
        }
