# MISpace Hackathon – Great Lakes Ice Forecasting and Data Science Platform  
### Link: https://elijahleemorgan.github.io/MISpaceHackathon/

A data science and machine learning project focused on Great Lakes ice forecasting using NOAA NetCDF (.nc) datasets. Includes full data processing pipelines, U-Net forecasting, visualization tools, and a static GitHub Pages dashboard.  
Built by students at Grand Valley State University who live, study, and work near the Great Lakes.

---

## 👥 Team Members and Roles

- **Marcos Sanson** – Machine Learning Model Development  
  Processed NetCDF data, built the forecasting dataset, developed and trained the U-Net model, and generated all HD visualizations and February predictions.

- **Elijah Morgan** – Data Reading & Analysis  
  Handled ingestion of raw NetCDF files, verified dataset structure and metadata, and organized the initial file system used across the project.

- **Darren Fife** – Data Reading & Documentation  
  Evaluated and interpreted the GLSEA dataset, confirmed the continuity of observations, and contributed to the end-to-end data preparation workflow.

- **Diego de Jong** – Frontend & Dashboard Development  
  Built the GitHub Pages site, designed the layout and visual presentation of predictions, and created the project demo video.

---

## ✨ Features

- **End-to-end pipeline** for Great Lakes ice concentration forecasting  
- **NetCDF ingestion and visualization tools** for NOAA GLSEA data  
- **Daily visualization generation** with stable color scales and custom land shading  
- **Machine learning dataset builder** using 7-day input windows  
- **U-Net model training** for next-day ice forecasting  
- **Automated February prediction generator** with versioned output folders and GIFs  
- **Static dashboard website** hosted through GitHub Pages  
- **Jupyter notebooks** for exploration and development  
- **HD visualization script** with grid overlays for targeted ice-clearing operations  
- **Clear, modular project structure** suitable for future extensions

---

## 📁 Project Structure

```
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
│   ├── daily_visualizations/         # Jan11–Jan31 PNGs
│   │
│   ├── data_processing/
│   │   ├── load_and_visualize.py
│   │   ├── inspect_nc.py
│   │   ├── downsample_data.py
│   │   ├── processor.py
│   │   └── nc_visualizer_outputs/
│   │
│   ├── models/
│   │   ├── predict_unet.py
│   │   ├── train_unet.py
│   │   └── checkpoints/
│   │
│   ├── utils/
│   │   └── gif_utils.py (optional)
│   │
│   └── predictions_ver_*/            # Feb predictions + GIFs
│
├── index.html
├── data-science.html
└── README.md
```

---

## 🌐 Viewing the Website

### GitHub Pages

[Live site](https://elijahleemorgan.github.io/MiSpaceHackathon/)

### Local Development

```bash
git clone https://github.com/ElijahLeeMorgan/MISpaceHackathon.git
cd MISpaceHackathon
```

Open `index.html` or run:

```bash
python -m http.server 8000
```

Visit: `http://localhost:8000`

---

## 👨‍💻 Python Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚦 Quick Start

### Viewing the Dashboard

Open the website or load `index.html`.

### Working with Weather Data

```bash
pip install xarray netCDF4 pandas numpy matplotlib
```

```python
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset('weather_data.nc')
print(ds)

temp = ds['temperature']
temp.mean(dim=['lat', 'lon']).plot()
plt.title('Average Temperature Over Time')
plt.show()
```

---

## 📊 Working with NetCDF Files

### What are NetCDF Files?

NetCDF (Network Common Data Form) files are widely used in climate and weather science. They store large multi-dimensional arrays such as temperature, ice concentration, wind speed, and more.

### Useful Libraries

```bash
pip install xarray netCDF4 numpy pandas matplotlib cartopy
```

---

## 🤖 Machine Learning Workflow

### Dataset Construction

The forecasting dataset converts daily 1024×1024 GLSEA maps into:

* **Inputs:** 7 consecutive days
* **Label:** next-day ice map

Resulting shapes:

```
X: (N, 7, 1024, 1024)
y: (N, 1024, 1024)
```

### Training the U-Net

```bash
python src/models/train_unet.py
```

* Trains for 5 epochs
* Saves checkpoints to `src/models/checkpoints/`

### Generating February Predictions

```bash
python src/models/predict_unet.py
```

Outputs:

```
predictions_ver_1/
  Feb01.png
  Feb02.png
  Feb03.png
  Feb04.png
  forecast.gif
```

Enhancements include:

* Clamp to 0–6 range
* Stabilization smoothing
* Light-blue land shading
* HD rendering and GIF generation

---

## 📓 Jupyter Notebooks

`notebooks/01_data_exploration.ipynb` demonstrates:

* NetCDF inspection
* Visualization of Great Lakes patterns
* Data cleaning
* Early ML experiments

---

## 🌍 Data Sources

* NOAA GLSEA (Great Lakes Surface Environmental Analysis)
* NOAA CoastWatch / GLERL
* NCEP / HRRR / GFS weather model fields (optional future extensions)

---

## 🤝 Contributing

1. Fork
2. Create a feature branch
3. Implement changes
4. Submit a pull request

Possible contribution areas:

* Better visualization
* Additional preprocessing
* New ML architectures
* More interactive dashboard features

---

## 📜 License

MIT License — see `LICENSE`.

---

## 🧑‍💻 Authors

* Marcos Sanson
* Elijah Lee Morgan
* Diego De Jong
* Darren Fife

---

## 🙏 Acknowledgments

* xarray and NetCDF communities
* NOAA, NASA, ECMWF for open data
* Python data science community
* PyTorch and scikit-learn teams
* GitHub Pages for hosting

---

Built with ❤️ by students at **Grand Valley State University**, inspired by the Great Lakes we live beside and study around.
