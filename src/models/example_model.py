"""Example machine learning model."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Any
import joblib
from pathlib import Path


class ExampleModel:
    """Example ML model for demonstration."""
    
    def __init__(self):
        """Initialize the model."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model.
        
        Args:
            X: Training features
            y: Training labels
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: List[float]) -> Any:
        """Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Model predictions
        """
        if not self.is_trained:
            # For demo purposes, return mock prediction
            return {
                "class": np.random.randint(0, 2),
                "confidence": np.random.random()
            }
        
        X_array = np.array(X).reshape(1, -1)
        X_scaled = self.scaler.transform(X_array)
        prediction = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return {
            "class": int(prediction[0]),
            "probabilities": probabilities[0].tolist()
        }
    
    def save(self, path: str):
        """Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str):
        """Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
