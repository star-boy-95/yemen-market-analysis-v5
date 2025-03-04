"""
Setup script for yemen-market-integration package.
"""
from setuptools import setup, find_packages

setup(
    name="yemen-market-integration",
    version="0.1",
    packages=find_packages(),
    install_requires=[
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
        "esda>=2.4.1"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Econometric analysis of market integration in Yemen",
    keywords="market integration, conflict economics, threshold cointegration, spatial econometrics",
    python_requires=">=3.8",
)