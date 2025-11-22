"""Generate sample data for testing and demonstration."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

# Create data directory if it doesn't exist
data_dir = Path(__file__).parent.parent / "data"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"

raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)


def generate_sample_csv():
    """Generate sample CSV data."""
    n_samples = 1000
    
    data = pd.DataFrame({
        'id': range(1, n_samples + 1),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.exponential(2, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })
    
    # Add some missing values
    data.loc[np.random.choice(data.index, 50, replace=False), 'feature1'] = np.nan
    data.loc[np.random.choice(data.index, 30, replace=False), 'feature2'] = np.nan
    
    # Save to CSV
    csv_path = raw_dir / "sample_data.csv"
    data.to_csv(csv_path, index=False)
    print(f"Generated sample CSV: {csv_path}")
    
    return data


def generate_sample_json():
    """Generate sample JSON data."""
    data = {
        "metadata": {
            "version": "1.0",
            "created": "2024-01-01",
            "description": "Sample data for MISpace Hackathon"
        },
        "records": [
            {
                "id": i,
                "value": float(np.random.randn()),
                "category": np.random.choice(['X', 'Y', 'Z']),
                "metrics": {
                    "score": float(np.random.uniform(0, 100)),
                    "confidence": float(np.random.uniform(0, 1))
                }
            }
            for i in range(100)
        ]
    }
    
    # Save to JSON
    json_path = raw_dir / "sample_data.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Generated sample JSON: {json_path}")


def generate_processed_data(raw_data):
    """Generate processed data example."""
    # Simple processing: remove missing values and normalize
    processed = raw_data.dropna()
    
    # Normalize numeric features
    numeric_cols = ['feature1', 'feature2', 'feature3', 'feature4']
    for col in numeric_cols:
        processed[col] = (processed[col] - processed[col].mean()) / processed[col].std()
    
    # Save to processed directory
    processed_path = processed_dir / "processed_data.csv"
    processed.to_csv(processed_path, index=False)
    print(f"Generated processed CSV: {processed_path}")


def main():
    """Generate all sample data."""
    print("Generating sample data...")
    
    # Generate raw data
    raw_data = generate_sample_csv()
    generate_sample_json()
    
    # Generate processed data
    generate_processed_data(raw_data)
    
    print("\nSample data generation complete!")
    print(f"Raw data: {raw_dir}")
    print(f"Processed data: {processed_dir}")


if __name__ == "__main__":
    main()
