"""
Setup script for yemen-market-integration package.

This package implements econometric methodologies for analyzing market integration
in conflict-affected Yemen, with a focus on threshold cointegration and 
spatial econometric techniques.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="yemen-market-integration",
    version="0.5.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    author="Yemen Market Integration Team",
    author_email="malakkaoui@worldbank.org",
    description="Econometric analysis of market integration in Yemen",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="market integration, conflict economics, threshold cointegration, spatial econometrics",
    url="https://github.com/star-boy-95/yemen-market-analysis-v5",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    entry_points={
        "console_scripts": [
            "yemen-analysis=src.main:main",
        ],
    },
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "geopandas>=0.10.0",
        "pyproj>=3.0.0",
        "pysal>=2.4.0",
        "folium>=0.12.0",
        "contextily>=1.1.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "arch>=5.0.0",
        "pmdarima>=1.8.0",
        "spreg>=1.2.4",
        "libpysal>=4.5.0",
        "esda>=2.4.1",
        "splot>=1.1.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.3.0",
        "mapclassify>=2.4.0",
        "numba>=0.55.0",
        "joblib>=1.1.0",
        "dask>=2022.1.0",
        "swifter>=1.0.0",
        "requests>=2.27.0",
        "beautifulsoup4>=4.10.0",
        "openpyxl>=3.0.0",
        "xlrd>=2.0.0",
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
        "flake8>=3.9.0",
        "black>=21.5b2",
        "pylint>=2.8.0",
        "sphinx>=4.3.0",
        "sphinx-rtd-theme>=1.0.0",
        "nbsphinx>=0.8.0",
        "pre-commit>=2.17.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
    ],
    include_package_data=True,
    package_data={
        "yemen_market_integration": [
            "config/*.yaml",
            "data/processed/*.csv",
            "data/processed/*.geojson",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "flake8>=3.9.0",
            "black>=21.5b2",
            "pylint>=2.8.0",
            "sphinx>=4.3.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.3.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
    },
)
