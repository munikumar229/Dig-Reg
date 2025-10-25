"""
API endpoints for digit classification
"""
import numpy as np
from fastapi import APIRouter, HTTPException
from ..models.loader import ModelLoader
from .schemas import (
    PredictionRequest, 
    PredictionResponse, 
    HealthResponse, 
    ModelsResponse,
    ErrorResponse
)

router = APIRouter()

# Initialize model loader
model_loader = ModelLoader()


@router.get("/", response_model=dict)
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Digits Classification API!",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predict digit from pixel values",
            "/health": "GET - Health check",
            "/models": "GET - List available models"
        }
    }


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="digits-classification-api"
    )


@router.get("/models", response_model=ModelsResponse)
def list_models():
    """Get available models."""
    return ModelsResponse(
        available_models=model_loader.get_available_models(),
        default_model="randomforest"
    )


@router.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def predict_digit(request: PredictionRequest):
    """Predict digit from pixel values."""
    try:
        # Validate input
        if len(request.pixels) != 64:
            raise HTTPException(
                status_code=400, 
                detail="Expected 64 pixel values (8x8 image)"
            )
        
        model_type = request.model_type.lower()
        if model_type not in ["randomforest", "mlp"]:
            raise HTTPException(
                status_code=400, 
                detail="Model type must be 'randomforest' or 'mlp'"
            )
        
        # Load model (with caching)
        model = model_loader.load_model_by_type(model_type)
        
        # Prepare input data
        input_array = np.array(request.pixels).reshape(1, -1)
        
        # Scale data for MLP (if needed)
        # Note: For simplicity, we're using a basic scaler here
        # In production, you should save and load the exact scaler used during training
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        probabilities = model.predict_proba(input_array)[0].tolist()
        confidence = float(max(probabilities))
        
        return PredictionResponse(
            predicted_digit=int(prediction),
            confidence=confidence,
            probabilities=probabilities,
            model_used=model_type
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )