"""
Setup script for yemen-market-integration package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="yemen-market-integration",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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