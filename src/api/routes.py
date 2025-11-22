"""API routes and endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from src.models.example_model import ExampleModel
from src.data_processing.processor import DataProcessor

router = APIRouter(prefix="/api", tags=["api"])

# Initialize model and processor
example_model = ExampleModel()
data_processor = DataProcessor()


@router.get("/status")
async def get_status():
    """Get API status and information."""
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0"
    }


@router.get("/data/summary")
async def get_data_summary():
    """Get summary statistics of the dataset."""
    try:
        # Example: Generate sample data
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
        })
        
        summary = data.describe().to_dict()
        
        return {
            "summary": summary,
            "shape": data.shape,
            "columns": list(data.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def make_prediction(data: Dict[str, Any]):
    """Make predictions using the model."""
    try:
        features = data.get("features", [])
        if not features:
            raise HTTPException(status_code=400, detail="No features provided")
        
        prediction = example_model.predict(features)
        
        return {
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualizations/sample")
async def get_sample_visualization():
    """Get sample data for visualization."""
    try:
        # Generate sample time series data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        data = {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'values': np.cumsum(np.random.randn(30)).tolist(),
            'category_a': np.random.randint(10, 100, 30).tolist(),
            'category_b': np.random.randint(10, 100, 30).tolist(),
        }
        
        return {
            "data": data,
            "metadata": {
                "type": "time_series",
                "start_date": data['dates'][0],
                "end_date": data['dates'][-1]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/process")
async def process_data(config: Dict[str, Any]):
    """Process data with given configuration."""
    try:
        result = data_processor.process(config)
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
