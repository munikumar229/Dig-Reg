"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request model for digit prediction."""
    pixels: List[float] = Field(
        ..., 
        description="64 pixel values (8x8 image flattened)",
        min_items=64,
        max_items=64
    )
    model_type: Optional[str] = Field(
        default="randomforest",
        description="Type of model to use for prediction",
        pattern="^(randomforest|mlp)$"
    )


class PredictionResponse(BaseModel):
    """Response model for digit prediction."""
    predicted_digit: int = Field(..., description="Predicted digit (0-9)")
    confidence: float = Field(..., description="Confidence score of the prediction")
    probabilities: List[float] = Field(..., description="Probability for each digit class (0-9)")
    model_used: str = Field(..., description="Model type used for prediction")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")


class ModelsResponse(BaseModel):
    """Available models response model."""
    available_models: List[str] = Field(..., description="List of available model types")
    default_model: str = Field(..., description="Default model type")


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error message")