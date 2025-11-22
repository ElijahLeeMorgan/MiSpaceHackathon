# MISpace Hackathon - Data Science Project

A comprehensive Python data science project template with FastAPI backend, interactive dashboard, and machine learning capabilities.

## 🚀 Features

- **FastAPI Backend**: RESTful API with automatic documentation
- **Interactive Dashboard**: Real-time data visualization and analytics
- **Data Processing Pipeline**: Robust data cleaning and transformation utilities
- **Machine Learning Models**: Example ML models with training and prediction
- **Jupyter Notebooks**: Interactive data exploration and analysis
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Testing Suite**: Comprehensive unit tests with pytest
- **Code Quality**: Linting, formatting, and type checking tools

## 📁 Project Structure

```
MISpaceHackathon/
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py        # Application settings
├── data/                  # Data storage
│   ├── raw/              # Raw data files
│   ├── processed/        # Processed data files
│   └── external/         # External data sources
├── frontend/             # Dashboard frontend
│   ├── static/
│   │   ├── css/         # Stylesheets
│   │   └── js/          # JavaScript files
│   └── templates/       # HTML templates
│       └── index.html   # Main dashboard
├── notebooks/            # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── scripts/              # Utility scripts
│   └── generate_sample_data.py
├── src/                  # Source code
│   ├── api/             # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py      # API entry point
│   │   └── routes.py    # API endpoints
│   ├── data_processing/ # Data processing utilities
│   │   ├── __init__.py
│   │   └── processor.py
│   ├── models/          # Machine learning models
│   │   ├── __init__.py
│   │   └── example_model.py
│   └── utils/           # Helper functions
│       ├── __init__.py
│       └── helpers.py
├── tests/               # Test suite
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data_processing.py
│   └── test_models.py
├── .env.example         # Environment variables template
├── .gitignore          # Git ignore rules
├── docker-compose.yml  # Docker Compose configuration
├── Dockerfile          # Docker image definition
├── Makefile            # Build automation
├── requirements.txt    # Python dependencies
└── setup.py            # Package setup

```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Docker and Docker Compose

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ElijahLeeMorgan/MISpaceHackathon.git
   cd MISpaceHackathon
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   make install
   # Or directly with pip:
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Generate sample data** (optional):
   ```bash
   python scripts/generate_sample_data.py
   ```

### Docker Installation

1. **Build and start containers**:
   ```bash
   make docker-build
   make docker-up
   ```

2. **Access the services**:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Jupyter: http://localhost:8888

## 🚦 Quick Start

### Running the API Server

**Using Make**:
```bash
make start
```

**Direct Python**:
```bash
python -m src.api.main
```

**With custom configuration**:
```bash
API_PORT=8080 python -m src.api.main
```

The API will be available at:
- Dashboard: http://localhost:8000
- Interactive API Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
pytest tests/test_api.py -v
```

### Code Quality

```bash
# Lint code
make lint

# Format code
make format

# Type checking
make typecheck
```

## 📊 API Endpoints

### Health & Status

- `GET /health` - Health check
- `GET /api/status` - API status and information

### Data Operations

- `GET /api/data/summary` - Get data summary statistics
- `POST /api/data/process` - Process data with configuration

### Machine Learning

- `POST /api/predict` - Make predictions with the model

### Visualizations

- `GET /api/visualizations/sample` - Get sample visualization data

## 🎯 Usage Examples

### Making Predictions

```python
import requests

url = "http://localhost:8000/api/predict"
data = {
    "features": [0.5, -0.3, 0.8]
}

response = requests.post(url, json=data)
print(response.json())
```

### Getting Data Summary

```python
import requests

url = "http://localhost:8000/api/data/summary"
response = requests.get(url)
summary = response.json()
print(summary)
```

### Using the Data Processor

```python
from src.data_processing.processor import DataProcessor
import pandas as pd

# Initialize processor
processor = DataProcessor()

# Load data
data = processor.load_data("data/raw/sample_data.csv")

# Clean data
cleaned = processor.clean_data()

# Get statistics
stats = processor.get_statistics()
```

### Training a Model

```python
from src.models.example_model import ExampleModel
import numpy as np

# Create sample data
X = np.random.randn(100, 3)
y = np.random.randint(0, 2, 100)

# Initialize and train model
model = ExampleModel()
model.train(X, y)

# Make prediction
prediction = model.predict([0.5, -0.3, 0.8])
print(prediction)
```

## 📓 Jupyter Notebooks

Start Jupyter to explore the notebooks:

```bash
jupyter notebook notebooks/
```

The included notebook demonstrates:
- Data loading and exploration
- Visualization techniques
- Data processing pipelines
- Model training and prediction

## 🐳 Docker Commands

```bash
# Build images
make docker-build

# Start services
make docker-up

# Stop services
make docker-down

# View logs
make docker-logs
```

## 🧪 Development

### Project Setup for Development

1. Install development dependencies:
   ```bash
   make install-dev
   ```

2. Run tests before committing:
   ```bash
   make test
   ```

3. Format and lint your code:
   ```bash
   make format
   make lint
   ```

### Adding New Features

1. Create feature branch
2. Add your code in appropriate directories
3. Write tests in `tests/`
4. Update documentation
5. Run tests and linting
6. Submit pull request

## 📝 Configuration

Configuration is managed through environment variables. See `.env.example` for available options:

- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `DATABASE_URL`: Database connection string
- `ENVIRONMENT`: Environment (development/production)
- `LOG_LEVEL`: Logging level (INFO/DEBUG/WARNING/ERROR)

## 🔒 Security

- Never commit `.env` file with secrets
- Change `SECRET_KEY` in production
- Use environment-specific configurations
- Keep dependencies updated

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- MISpace Hackathon Team

## 🙏 Acknowledgments

- FastAPI for the amazing web framework
- Scikit-learn for machine learning capabilities
- Plotly and Chart.js for visualizations
- The open-source community

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review the example notebooks

---

Built with ❤️ for the MISpace Hackathon