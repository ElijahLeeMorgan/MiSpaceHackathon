# MISpace Hackathon - Weather Data Science Dashboard

A static GitHub Pages website and dashboard for weather data science and machine learning projects. Focus on working with NetCDF (.nc) files and weather data analysis.

## 🌦️ Features

- **Static Dashboard**: GitHub Pages hosted website for project documentation
- **Weather Data Focus**: Specialized in NetCDF (.nc) file processing and analysis
- **Data Science Section**: Comprehensive guide for ML work with weather data
- **Machine Learning**: Examples and guides for weather prediction and pattern recognition
- **Jupyter Notebooks**: Interactive data exploration and analysis
- **Simple & Clean**: Easy to navigate interface without backend dependencies

## 📁 Project Structure

```
MISpaceHackathon/
├── assets/                 # Static website assets
│   └── css/               # Stylesheets
│       └── style.css
├── data/                  # Weather data storage
│   ├── raw/              # Raw NetCDF files
│   ├── processed/        # Processed datasets
│   └── external/         # External data sources
├── notebooks/            # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── src/                  # Python source code
│   ├── data_processing/ # Data processing utilities
│   ├── models/          # ML models
│   └── utils/           # Helper functions
├── index.html           # Main landing page
├── data-science.html    # Data science guide page
└── README.md            # This file

```

## 🛠️ Viewing the Website

### GitHub Pages

Visit the live website at: `https://elijahleemorgan.github.io/MISpaceHackathon/`

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ElijahLeeMorgan/MISpaceHackathon.git
   cd MISpaceHackathon
   ```

2. **Open in browser**:
   Simply open `index.html` in your web browser, or use a simple HTTP server:
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Then visit: http://localhost:8000
   ```

3. **For Python Data Science Work**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

## 🚦 Quick Start

### Viewing the Dashboard

Simply open the website in your browser or visit the GitHub Pages URL.

### Working with Weather Data

```python
# Install required libraries
pip install xarray netCDF4 pandas numpy matplotlib

# Example: Load and analyze a NetCDF file
import xarray as xr
import matplotlib.pyplot as plt

# Open NetCDF file
ds = xr.open_dataset('weather_data.nc')

# View dataset info
print(ds)

# Access temperature data
temp = ds['temperature']

# Plot average temperature
temp.mean(dim=['lat', 'lon']).plot()
plt.title('Average Temperature Over Time')
plt.show()
```

### Running Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab
```

## 📊 Working with NetCDF Files

### What are NetCDF Files?

NetCDF (Network Common Data Form) files are the standard format for weather and climate data:
- Multi-dimensional arrays (time, latitude, longitude, altitude)
- Self-describing with embedded metadata
- Efficient storage and access
- Widely used in meteorology and climate science

### Essential Python Libraries

```bash
pip install xarray netCDF4 numpy pandas matplotlib cartopy
```

### Common Weather Data Variables

- **Temperature**: Surface and atmospheric temperature
- **Precipitation**: Rainfall, snowfall amounts
- **Wind**: Speed, direction, gusts
- **Pressure**: Atmospheric and sea-level pressure
- **Humidity**: Relative and specific humidity
- **Cloud Cover**: Cloud fraction and types

## 🎯 Machine Learning Applications

### Weather Forecasting

Predict future weather conditions using:
- Time series models (ARIMA, LSTM)
- Random Forests and Gradient Boosting
- Neural Networks

### Pattern Recognition

Identify weather patterns and anomalies:
- Clustering algorithms (K-means, DBSCAN)
- Classification models
- Anomaly detection

### Example ML Workflow

```python
from sklearn.ensemble import RandomForestRegressor
import xarray as xr
import pandas as pd

# Load weather data
ds = xr.open_dataset('weather_data.nc')

# Prepare features
features = ['temperature', 'pressure', 'humidity']
X = pd.DataFrame({feat: ds[feat].values.flatten() for feat in features})
y = ds['future_temp'].values.flatten()

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Make predictions
predictions = model.predict(X_test)
```

## 📓 Jupyter Notebooks

Explore data interactively:

```bash
# Start Jupyter Notebook
jupyter notebook notebooks/

# Or JupyterLab
jupyter lab
```

Example notebooks included:
- Data exploration with NetCDF files
- Weather data visualization
- Machine learning for weather prediction
- Time series analysis

## 🧪 Development

### Project Setup

1. Clone and navigate to the repository
2. Install Python dependencies: `pip install -r requirements.txt`
3. Open `index.html` in a browser or use a local server
4. Modify HTML/CSS in `assets/` directory
5. Add Python code in `src/` directory for data processing

### Adding New Content

1. Create new HTML pages as needed
2. Link them in the navigation
3. Style with existing CSS or add custom styles
4. Keep it simple and focused on weather data science

## 📝 Weather Data Sources

### Recommended Data Sources

- **NOAA**: National Oceanic and Atmospheric Administration
  - https://www.noaa.gov
  - Free weather and climate data

- **ECMWF**: European Centre for Medium-Range Weather Forecasts
  - https://www.ecmwf.int
  - Reanalysis data (ERA5)

- **NASA**: Earth Observing System
  - https://earthdata.nasa.gov
  - Satellite weather data

- **NCEP**: National Centers for Environmental Prediction
  - Numerical weather prediction data

### Python Libraries for Weather Data

```bash
# Core libraries
pip install xarray netCDF4 numpy pandas

# Visualization
pip install matplotlib seaborn cartopy

# Machine Learning
pip install scikit-learn tensorflow pytorch

# Additional tools
pip install scipy jupyter
```

## 🤝 Contributing

Contributions welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes (HTML, CSS, Python code)
4. Test your changes locally
5. Submit a pull request

Focus areas:
- Weather data analysis examples
- ML model implementations
- Data visualization improvements
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- MISpace Hackathon Team

## 🙏 Acknowledgments

- xarray and NetCDF communities for excellent tools
- NOAA, ECMWF, and NASA for providing open weather data
- Python data science community
- Scikit-learn and TensorFlow teams
- GitHub Pages for free hosting

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the Data Science guide page
- Review example notebooks
- Visit weather data source documentation

---

Built with ❤️ for weather data science and machine learning