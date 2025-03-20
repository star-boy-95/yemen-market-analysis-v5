#!/bin/bash
# Build script for building the Python package

set -e  # Exit immediately if a command fails

LOG_FILE="build.log"
echo "Starting build process at $(date)" > "$LOG_FILE"

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv venv >> "$LOG_FILE" 2>&1 || { echo "Error: Failed to create virtual environment"; exit 1; }

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate >> "$LOG_FILE" 2>&1 || { echo "Error: Failed to activate virtual environment"; exit 1; }

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt >> "$LOG_FILE" 2>&1 || { echo "Error: Failed to install dependencies"; exit 1; }

# Build the package
echo "Building the package..."
python setup.py sdist >> "$LOG_FILE" 2>&1 || { echo "Error: Failed to build the package"; exit 1; }

# Install the package
echo "Installing the package..."
pip install dist/yemen-market-integration-0.1.0.tar.gz >> "$LOG_FILE" 2>&1 || { echo "Error: Failed to install the package"; exit 1; }

# Run tests
echo "Running tests with pytest..."
pytest >> "$LOG_FILE" 2>&1 || { echo "Error: Tests failed"; exit 1; }

# Deactivate the virtual environment
echo "Deactivating virtual environment..."
deactivate >> "$LOG_FILE" 2>&1 || { echo "Error: Failed to deactivate virtual environment"; exit 1; }

# Remove the virtual environment and dist directory
echo "Removing virtual environment and dist directory..."
rm -rf venv dist >> "$LOG_FILE" 2>&1 || { echo "Error: Failed to remove virtual environment and dist directory"; exit 1; }

echo "Build complete. Check $LOG_FILE for details."
