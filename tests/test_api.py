"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "version" in data


def test_api_status():
    """Test API status endpoint."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data


def test_data_summary():
    """Test data summary endpoint."""
    response = client.get("/api/data/summary")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "shape" in data
    assert "columns" in data


def test_predict():
    """Test prediction endpoint."""
    payload = {
        "features": [0.5, -0.3, 0.8]
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "timestamp" in data


def test_predict_no_features():
    """Test prediction endpoint with no features."""
    payload = {}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400


def test_visualizations():
    """Test visualizations endpoint."""
    response = client.get("/api/visualizations/sample")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "metadata" in data


def test_process_data():
    """Test data processing endpoint."""
    payload = {
        "operation": "summary"
    }
    response = client.post("/api/data/process", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"
