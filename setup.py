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
    # Load requirements from requirements.txt
    install_requires=open("requirements.txt").read().strip().split("\n"),
    include_package_data=True,
    package_data={
        "yemen_market_integration": [
            "config/*.yaml",
            "data/processed/*.csv",
            "data/processed/*.geojson",
        ],
    },
    # Load development requirements from requirements-dev.txt
    extras_require={
        "dev": [line.strip() for line in open("requirements-dev.txt").readlines() 
               if not line.startswith("-r") and not line.startswith("#") and line.strip()]
    },
)
