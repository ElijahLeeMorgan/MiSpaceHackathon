"""Tests for machine learning models."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.example_model import ExampleModel


@pytest.fixture
def model():
    """Create a model instance."""
    return ExampleModel()


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randint(0, 2, 100)
    return X, y


def test_model_initialization(model):
    """Test model initialization."""
    assert model is not None
    assert not model.is_trained


def test_model_train(model, sample_data):
    """Test model training."""
    X, y = sample_data
    model.train(X, y)
    assert model.is_trained


def test_model_predict_untrained(model):
    """Test prediction with untrained model."""
    features = [0.5, -0.3, 0.8]
    prediction = model.predict(features)
    assert prediction is not None
    assert "class" in prediction


def test_model_predict_trained(model, sample_data):
    """Test prediction with trained model."""
    X, y = sample_data
    model.train(X, y)
    
    features = [0.5, -0.3, 0.8]
    prediction = model.predict(features)
    
    assert prediction is not None
    assert "class" in prediction
    assert "probabilities" in prediction
    assert isinstance(prediction["class"], int)
    assert isinstance(prediction["probabilities"], list)


def test_model_save_load(model, sample_data, tmp_path):
    """Test model save and load."""
    X, y = sample_data
    model.train(X, y)
    
    # Save model
    model_path = tmp_path / "test_model.pkl"
    model.save(str(model_path))
    assert model_path.exists()
    
    # Load model
    new_model = ExampleModel()
    new_model.load(str(model_path))
    assert new_model.is_trained
    
    # Test prediction
    features = [0.5, -0.3, 0.8]
    prediction = new_model.predict(features)
    assert prediction is not None
