# Data Directory

This directory contains all project data files organized into subdirectories:

## Structure

- **raw/**: Original, immutable data files
- **processed/**: Cleaned and transformed data ready for analysis
- **external/**: Data from external sources or third-party APIs

## Usage

Place your raw data files in the `raw/` directory and use the data processing utilities to generate cleaned versions in `processed/`.

## Sample Data

Generate sample data using:
```bash
python scripts/generate_sample_data.py
```

This will create example datasets for testing and development.
