from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mi_space_hackathon",
    version="0.1.0",
    author="MISpace Hackathon Team",
    description="A Python data science project with web server backend and dashboard",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ElijahLeeMorgan/MISpaceHackathon",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "plotly>=5.17.0",
        "scipy>=1.11.2",
        "joblib>=1.3.2",
        "sqlalchemy>=2.0.21",
        "psycopg2-binary>=2.9.7",
        "pydantic>=2.4.2",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0",
            "httpx>=0.25.0",
            "black>=23.9.1",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
        ],
    },
)
