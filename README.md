Link: https://elijahleemorgan.github.io/MiSpaceHackathon/index.html

# MISpace Hackathon - Great Lakes Ice Forecasting and Data Science Platform

A data science and machine learning project focused on Great Lakes ice forecasting using NOAA NetCDF (.nc) datasets. Includes full data processing pipelines, U-Net forecasting, visualization tools, and a static GitHub Pages dashboard.

## Features

- **End-to-end pipeline** for Great Lakes ice concentration forecasting
- **NetCDF ingestion and visualization tools** for NOAA GLSEA data
- **Daily visualization generation** with stable color scales and custom land shading
- **Machine learning dataset builder** using 7-day input windows
- **U-Net model training** for next-day ice forecasting
- **Automated February prediction generator** with versioned output folders and GIFs
- **Static dashboard website** hosted through GitHub Pages
- **Jupyter notebooks** for exploration and development

## Project Structure

````markdown
MISpaceHackathon/
├── assets/
│   └── css/
│       └── style.css
│
├── data/
│   ├── raw/                 # (optional; primary raw data lives outside repo)
│   ├── processed/           # ML-ready arrays (X.npy, y.npy)
│   └── external/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── daily_visualizations/         # Jan11-Jan31 PNGs
│   │
│   ├── data_processing/
│   │   ├── load_and_visualize.py        # Generate daily PNGs and cache raw arrays
│   │   ├── inspect_nc.py                # Examine structure and metadata of NetCDF files
│   │   ├── downsample_data.py           # Create lower-resolution datasets for fast experiments
│   │   ├── processor.py                 # Utility class for NetCDF and shapefile preprocessing
│   │   └── nc_visualizer_outputs/       # Saved figures from netCDF visualization scripts
│   │
│   ├── models/
│   │   ├── predict_unet.py              # Run trained U-Net to produce February predictions and GIFs
│   │   ├── train_unet.py                # Train U-Net for 5 epochs using (X,y) processed arrays
│   │   └── checkpoints/                 # Model weights saved after each epoch
│   │
│   ├── utils/
│   │   └── gif_utils.py (optional)
│   │
│   └── predictions_ver_*/            # Feb predictions + GIFs
│
├── index.html
├── data-science.html
└── README.md
````

## Viewing the Website

### GitHub Pages

Live site:  
`https://elijahleemorgan.github.io/MISpaceHackathon/`

### Local Development

```bash
git clone https://github.com/ElijahLeeMorgan/MISpaceHackathon.git
cd MISpaceHackathon
```

Open `index.html` directly, or run:

```bash
python -m http.server 8000
```

## Local Development

### Clone the Repository
```bash
git clone https://github.com/ElijahLeeMorgan/MISpaceHackathon.git
cd MISpaceHackathon
```

### Serve the Site
Open `index.html` directly, or run:
```bash
python -m http.server 8000
```
Then visit `http://localhost:8000`.

### Python Data Science Environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
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


```bash
pip install xarray netCDF4 numpy pandas matplotlib cartopy
```

## Machine Learning Workflow

### Dataset Construction

The pipeline converts daily 1024×1024 ice maps into sequences:

- Inputs: 7 consecutive days
- Label: next day

Output shapes:

```
X: (N, 7, 1024, 1024)
y: (N, 1024, 1024)
```

### Training the U-Net

```bash
python src/models/train_unet.py
```

- Trains for 5 epochs
- Saves model checkpoints in `src/models/checkpoints/`

### Predicting Future Ice Maps (February)

```bash
python src/models/predict_unet.py
```

Outputs:

```
predictions_ver_1/
  feb01.png
  feb02.png
  ...
  animation.gif
```

The script applies:

- Value clipping (0-6)
- Median filtering
- Constant color scale
- Custom light-blue land shading

## Jupyter Notebooks

Interactive exploration available in `notebooks/01_data_exploration.ipynb`, covering:

- NetCDF structure
- Great Lakes spatial patterns
- Data cleaning and visualization
- Early forecasting experiments

## Data Sources

- **NOAA GLSEA** (Great Lakes Surface Environmental Analysis)
- **NOAA CoastWatch / GLERL**
- **NCEP/HRRR/GFS** for weather variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Submit a pull request

Areas for contribution:

- Visualization improvements
- Additional preprocessing steps
- New ML architectures
- Enhanced dashboard content

## License

MIT License. See `LICENSE` for details.

## Authors

- Marcos Sanson
- Elijah Lee Morgan
- Diego De Jong
- Darren Fife

## 🙏 Acknowledgments

- xarray and NetCDF communities for excellent tools
- NOAA, ECMWF, and NASA for providing open weather data
- Python data science community
- Scikit-learn and TensorFlow teams
- GitHub Pages for free hosting

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Review example notebooks
- Visit weather data source documentation

---

Built with ❤️ for weather data science and machine learning
