"""
Configuration and utilities for the Streamlit frontend
"""
import os
import requests
from typing import List


class APIClient:
    """Client for communicating with the FastAPI backend."""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000")
    
    def check_health(self) -> bool:
        """Check if the FastAPI backend is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from backend."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json().get("available_models", ["randomforest", "mlp"])
        except requests.exceptions.RequestException:
            pass
        return ["randomforest", "mlp"]
    
    def predict(self, pixels: List[float], model_type: str = "randomforest") -> dict:
        """Make prediction via FastAPI backend."""
        try:
            payload = {
                "pixels": pixels,
                "model_type": model_type
            }
            response = requests.post(
                f"{self.base_url}/predict", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error")
                raise Exception(f"API Error: {error_detail}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection Error: {str(e)}")


def preprocess_image_for_api(img):
    """Preprocess image for API prediction - normalize to 0-16 range and flatten."""
    import cv2
    from sklearn.preprocessing import MinMaxScaler
    
    img_resized = cv2.resize(img, (8,8), interpolation=cv2.INTER_AREA)
    img_resized = 255 - img_resized  # Invert colors (white background -> black)
    
    # Normalize to 0-16 range (similar to sklearn digits dataset)
    scaler_api = MinMaxScaler((0,16))
    img_normalized = scaler_api.fit_transform(img_resized)
    pixels = img_normalized.flatten().tolist()  # Convert to list for JSON
    
    return pixels, img_resized