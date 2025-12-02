"""Data processing utilities for netCDF and shapefile files."""

import pandas as pd
import numpy as np
import netCDF4 as nc
import geopandas as gpd
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


class DataProcessor:
    """Data processing and transformation utilities for netCDF and shapefile files."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.data = None
        self.dataset = None
        self.variables = {}
        self.geodata = None
        self.geometry_column = None
    
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
    
    def load_shapefile(self, filepath: str, attributes: Optional[List[str]] = None) -> pd.DataFrame:
        """Load data from a shapefile with weather-related geospatial data.
        
        This method loads shapefiles and prepares the data for ML model development
        by extracting geometric features, handling coordinate reference systems,
        and converting geospatial data to a format suitable for machine learning.
        
        Args:
            filepath: Path to the shapefile (.shp) file
            attributes: List of attribute names to load (loads all if None)
            
        Returns:
            DataFrame with extracted features ready for ML modeling
            
        Example:
            >>> processor = DataProcessor()
            >>> df = processor.load_shapefile('weather_stations.shp')
            >>> # Data includes geometric features (centroids, areas, etc.)
            >>> df = processor.prepare_shapefile_for_ml(df)
        """
        file_path = Path(filepath)
        
        if file_path.suffix != '.shp':
            raise ValueError(f"Expected .shp (shapefile) file, got: {file_path.suffix}")
        
        # Load shapefile using geopandas
        self.geodata = gpd.read_file(filepath)
        
        # Store the geometry column name
        self.geometry_column = self.geodata.geometry.name
        
        # Select specific attributes if provided
        if attributes is not None:
            available_attrs = [col for col in attributes if col in self.geodata.columns]
            if self.geometry_column not in available_attrs:
                available_attrs.append(self.geometry_column)
            self.geodata = self.geodata[available_attrs]
        
        # Extract geometric features for ML
        data = self._extract_geometric_features(self.geodata)
        
        # Store processed data
        self.data = data
        
        return self.data
    
    def _extract_geometric_features(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Extract ML-ready features from geometric data.
        
        Args:
            gdf: GeoDataFrame with geometric data
            
        Returns:
            DataFrame with extracted geometric features
        """
        df = gdf.copy()
        
        # Extract centroid coordinates (X, Y)
        df['centroid_x'] = df.geometry.centroid.x
        df['centroid_y'] = df.geometry.centroid.y
        
        # Extract bounds (bounding box coordinates)
        df['bounds_minx'] = df.geometry.bounds['minx']
        df['bounds_miny'] = df.geometry.bounds['miny']
        df['bounds_maxx'] = df.geometry.bounds['maxx']
        df['bounds_maxy'] = df.geometry.bounds['maxy']
        
        # Calculate geometric properties
        df['geometry_area'] = df.geometry.area
        df['geometry_length'] = df.geometry.length
        
        # Encode geometry type as categorical
        df['geometry_type'] = df.geometry.geom_type
        
        # Drop the original geometry column for ML (keep attributes)
        df = pd.DataFrame(df.drop(columns=[self.geometry_column]))
        
        return df
    
    def prepare_shapefile_for_ml(self, 
                                  data: Optional[pd.DataFrame] = None,
                                  encode_categorical: bool = True,
                                  normalize: bool = True,
                                  handle_missing: str = 'median') -> pd.DataFrame:
        """Prepare shapefile data for machine learning model development.
        
        This method applies comprehensive preprocessing including:
        - Handling missing values
        - Encoding categorical variables
        - Normalizing/standardizing numerical features
        - Removing highly correlated features
        
        Args:
            data: DataFrame to prepare (uses self.data if None)
            encode_categorical: Whether to one-hot encode categorical variables
            normalize: Whether to standardize numerical features
            handle_missing: Strategy for missing values ('median', 'mean', 'drop')
            
        Returns:
            ML-ready DataFrame
            
        Example:
            >>> processor = DataProcessor()
            >>> processor.load_shapefile('weather_data.shp')
            >>> ml_data = processor.prepare_shapefile_for_ml(
            ...     encode_categorical=True,
            ...     normalize=True
            ... )
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_shapefile first.")
            data = self.data.copy()
        else:
            data = data.copy()
        
        # Handle missing values
        if handle_missing == 'drop':
            data = data.dropna()
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if handle_missing == 'median':
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            elif handle_missing == 'mean':
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
            
            # Fill categorical missing values with mode
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
        
        # Encode categorical variables
        if encode_categorical:
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        
        # Normalize numerical features
        if normalize:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        # Remove duplicate rows
        data = data.drop_duplicates()
        
        return data
    
    def get_shapefile_metadata(self) -> Dict[str, Any]:
        """Get metadata from the loaded shapefile.
        
        Returns:
            Dictionary containing CRS, bounds, geometry types, and attributes
        """
        if self.geodata is None:
            return {}
        
        metadata = {
            "crs": str(self.geodata.crs),
            "total_bounds": self.geodata.total_bounds.tolist(),
            "geometry_types": self.geodata.geometry.geom_type.unique().tolist(),
            "num_features": len(self.geodata),
            "columns": list(self.geodata.columns),
            "geometry_column": self.geometry_column
        }
        
        return metadata
    
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
        """Close the netCDF dataset and clear geodata."""
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
        self.geodata = None
        self.geometry_column = None
    
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
