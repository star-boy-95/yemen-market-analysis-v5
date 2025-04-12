#!/usr/bin/env python
"""
Setup script for Yemen Market Analysis package.

This package provides tools for analyzing market integration in Yemen,
with a focus on econometric methods for time series and spatial analysis.
"""
from setuptools import setup, find_packages

setup(
    name="yemen_market_analysis",
    version="2.0.0",
    description="Econometric analysis tools for Yemen market integration",
    author="Mohammad Al-Akkaoui",
    author_email="mohammad@al-akkaoui.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "statsmodels>=0.13.0",
        "scipy>=1.7.0",
        "geopandas>=0.10.0",
        "plotly>=5.3.0",
        "dash>=2.0.0",
        "arch>=5.0.0",
        "ruptures>=1.1.5",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "mypy>=0.812",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "spatial": [
            "pysal>=2.3.0",
            "libpysal>=4.5.0",
            "esda>=2.4.0",
            "splot>=1.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
)
