"""
Setup script for yemen-market-integration package.
"""
from setuptools import setup, find_packages
import os
from pathlib import Path  # Fixed import from pathPath to pathlib

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Core dependencies
install_requires = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "geopandas>=0.10.0",
    "pysal>=2.4.0",
    "folium>=0.12.0",
    "statsmodels>=0.13.0",
    "arch>=5.0.0",
    "pmdarima>=1.8.0",
    "spreg>=1.2.4",
    "libpysal>=4.5.0",
    "plotly>=5.3.0",
    "mapclassify>=2.4.0",
    "contextily>=1.2.0",
    "esda>=2.4.1",
    "pyyaml>=6.0",
]

# Development dependencies
dev_requires = [
    "pytest>=6.2.0",
    "black>=21.5b2",
    "flake8>=3.9.0",
    "jupyter>=1.0.0",
    "sphinx>=4.0.0",
    "nbsphinx>=0.8.0",
    "pytest-cov>=2.12.0",
    "mypy>=0.910",
]

setup(
    name="yemen-market-integration",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
    },
    python_requires=">=3.8",
    author="Yemen Market Integration Team",
    author_email="your.email@example.com",
    description="Econometric analysis of market integration in Yemen",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="market integration, conflict economics, threshold cointegration, spatial econometrics",
    url="https://github.com/yourusername/yemen-market-integration",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
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
    include_package_data=True,
    package_data={
        "yemen_market_integration": ["config/*.yaml"],
    },
)