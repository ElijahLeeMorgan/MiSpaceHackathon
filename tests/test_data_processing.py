"""Tests for data processing module."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.processor import DataProcessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'feature3': ['a', 'b', 'c', 'd', 'e']
    })


@pytest.fixture
def processor():
    """Create a DataProcessor instance."""
    return DataProcessor()


def test_processor_initialization(processor):
    """Test processor initialization."""
    assert processor is not None
    assert processor.data is None


def test_clean_data(processor, sample_data):
    """Test data cleaning."""
    processor.data = sample_data
    cleaned = processor.clean_data()
    assert cleaned is not None
    assert isinstance(cleaned, pd.DataFrame)
    assert cleaned.shape == sample_data.shape


def test_get_statistics(processor, sample_data):
    """Test statistics generation."""
    processor.data = sample_data
    stats = processor.get_statistics()
    assert stats is not None
    assert "shape" in stats
    assert "columns" in stats
    assert "dtypes" in stats
    assert stats["shape"] == sample_data.shape


def test_transform_features(processor, sample_data):
    """Test feature transformation."""
    transformed = processor.transform_features(sample_data, method='standard')
    assert transformed is not None
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape == sample_data.shape


def test_process_config(processor):
    """Test process method with config."""
    config = {"operation": "summary"}
    result = processor.process(config)
    assert result is not None
    assert "status" in result
    assert result["status"] == "processed"
