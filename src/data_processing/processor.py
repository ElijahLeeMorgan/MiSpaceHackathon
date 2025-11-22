"""Data processing utilities for netCDF files."""

import pandas as pd
import numpy as np
import netCDF4 as nc
from typing import Dict, Any, List, Optional
from pathlib import Path


class DataProcessor:
    """Data processing and transformation utilities for netCDF files."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.data = None
        self.dataset = None
        self.variables = {}
    
    def load_data(self, filepath: str, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Load data from a netCDF file.
        
        Args:
            filepath: Path to the netCDF (.nc) file
            variables: List of variable names to load (loads all if None)
            
        Returns:
            Loaded DataFrame with variables as columns
        """
        file_path = Path(filepath)
        
        if file_path.suffix != '.nc':
            raise ValueError(f"Only .nc (netCDF) files are supported, got: {file_path.suffix}")
        
        # Open netCDF dataset
        self.dataset = nc.Dataset(filepath, 'r')
        
        # Extract variables
        data_dict = {}
        
        if variables is None:
            # Load all variables
            variables = [var for var in self.dataset.variables.keys()]
        
        for var_name in variables:
            if var_name in self.dataset.variables:
                var_data = self.dataset.variables[var_name][:]
                # Flatten if multi-dimensional
                if var_data.ndim > 1:
                    var_data = var_data.flatten()
                data_dict[var_name] = var_data
                self.variables[var_name] = self.dataset.variables[var_name]
        
        # Create DataFrame
        self.data = pd.DataFrame(data_dict)
        
        return self.data
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from the netCDF file.
        
        Returns:
            Dictionary containing dimensions, variables, and attributes
        """
        if self.dataset is None:
            return {}
        
        metadata = {
            "dimensions": {dim: len(self.dataset.dimensions[dim]) for dim in self.dataset.dimensions},
            "variables": list(self.dataset.variables.keys()),
            "global_attributes": {attr: self.dataset.getncattr(attr) for attr in self.dataset.ncattrs()}
        }
        
        return metadata
    
    def close(self):
        """Close the netCDF dataset."""
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
    
    def clean_data(self, data: pd.DataFrame = None, fill_value: Optional[float] = None) -> pd.DataFrame:
        """Clean the data by handling missing values and masked values from netCDF.
        
        Args:
            data: DataFrame to clean (uses self.data if None)
            fill_value: Value to use for filling missing data (uses median if None)
            
        Returns:
            Cleaned DataFrame
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data first.")
            data = self.data.copy()
        else:
            data = data.copy()
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values and masked arrays from netCDF
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if fill_value is not None:
            data[numeric_columns] = data[numeric_columns].fillna(fill_value)
        else:
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
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
