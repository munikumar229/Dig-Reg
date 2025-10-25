"""
Tests for the FastAPI backend
"""
import pytest
import requests
import numpy as np
from typing import Dict, Any


class TestBackendAPI:
    """Test suite for the backend API endpoints."""
    
    BASE_URL = "http://localhost:8000"
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "digits-classification-api"
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = requests.get(f"{self.BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_models_endpoint(self):
        """Test the models listing endpoint."""
        response = requests.get(f"{self.BASE_URL}/models")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "default_model" in data
        assert isinstance(data["available_models"], list)
        assert "randomforest" in data["available_models"]
        assert "mlp" in data["available_models"]
    
    def test_predict_endpoint_valid_input(self):
        """Test prediction with valid input."""
        # Create test data (64 pixel values)
        pixels = np.random.rand(64).tolist()
        payload = {
            "pixels": pixels,
            "model_type": "randomforest"
        }
        
        response = requests.post(f"{self.BASE_URL}/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_digit" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "model_used" in data
        
        assert 0 <= data["predicted_digit"] <= 9
        assert 0 <= data["confidence"] <= 1
        assert len(data["probabilities"]) == 10
        assert data["model_used"] == "randomforest"
    
    def test_predict_endpoint_invalid_pixels(self):
        """Test prediction with invalid pixel count."""
        # Wrong number of pixels (should be 64)
        pixels = [0.5] * 32
        payload = {
            "pixels": pixels,
            "model_type": "randomforest"
        }
        
        response = requests.post(f"{self.BASE_URL}/predict", json=payload)
        assert response.status_code == 400
        assert "Expected 64 pixel values" in response.json()["detail"]
    
    def test_predict_endpoint_invalid_model(self):
        """Test prediction with invalid model type."""
        pixels = [0.5] * 64
        payload = {
            "pixels": pixels,
            "model_type": "invalid_model"
        }
        
        response = requests.post(f"{self.BASE_URL}/predict", json=payload)
        assert response.status_code == 400
        assert "Model type must be" in response.json()["detail"]
    
    def test_predict_both_models(self):
        """Test prediction with both model types."""
        pixels = np.random.rand(64).tolist()
        
        for model_type in ["randomforest", "mlp"]:
            payload = {
                "pixels": pixels,
                "model_type": model_type
            }
            
            response = requests.post(f"{self.BASE_URL}/predict", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert data["model_used"] == model_type


if __name__ == "__main__":
    # Run basic tests if backend is running
    test_api = TestBackendAPI()
    
    try:
        print("Testing Backend API...")
        test_api.test_health_endpoint()
        print("✅ Health endpoint OK")
        
        test_api.test_models_endpoint()
        print("✅ Models endpoint OK")
        
        test_api.test_predict_endpoint_valid_input()
        print("✅ Prediction endpoint OK")
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure the backend is running: docker-compose up backend")