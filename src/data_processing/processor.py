"""Data processing utilities."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path


class DataProcessor:
    """Data processing and transformation utilities."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.data = None
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from a file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(filepath)
        
        if file_path.suffix == '.csv':
            self.data = pd.read_csv(filepath)
        elif file_path.suffix in ['.xlsx', '.xls']:
            self.data = pd.read_excel(filepath)
        elif file_path.suffix == '.json':
            self.data = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return self.data
    
    def clean_data(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Clean the data by handling missing values and outliers.
        
        Args:
            data: DataFrame to clean (uses self.data if None)
            
        Returns:
            Cleaned DataFrame
        """
        if data is None:
            data = self.data.copy()
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
        
        return data
    
    def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process data based on configuration.
        
        Args:
            config: Processing configuration
            
        Returns:
            Processing results
        """
        result = {
            "status": "processed",
            "config": config,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Example processing based on config
        if config.get("operation") == "summary":
            if self.data is not None:
                result["summary"] = self.data.describe().to_dict()
        
        return result
    
    def transform_features(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Transform features using specified method.
        
        Args:
            data: Input DataFrame
            method: Transformation method ('standard', 'minmax', 'log')
            
        Returns:
            Transformed DataFrame
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        transformed = data.copy()
        
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            transformed[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            transformed[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        elif method == 'log':
            transformed[numeric_columns] = np.log1p(data[numeric_columns])
        
        return transformed
    
    def get_statistics(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """Get statistical summary of the data.
        
        Args:
            data: DataFrame to analyze (uses self.data if None)
            
        Returns:
            Dictionary with statistics
        """
        if data is None:
            data = self.data
        
        if data is None:
            return {}
        
        return {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "summary_stats": data.describe().to_dict()
        }
