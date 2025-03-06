#!/bin/bash
# Build script for compiling the LaTeX document

set -e  # Exit immediately if a command fails

LOG_FILE="build.log"
echo "Starting build process at $(date)" > "$LOG_FILE"

# Move to the academic_paper directory
cd academic_paper || { echo "Error: academic_paper directory not found"; exit 1; }

# First run of XeLaTeX
echo "Running first XeLaTeX pass..."
xelatex -interaction=nonstopmode main.tex >> "$LOG_FILE" 2>&1 || { echo "Error: XeLaTeX first pass failed"; exit 1; }

# Run Biber for bibliography
echo "Running Biber..."
biber main >> "$LOG_FILE" 2>&1 || { echo "Error: Biber failed"; exit 1; }

# Second run of XeLaTeX
echo "Running second XeLaTeX pass..."
xelatex -interaction=nonstopmode main.tex >> "$LOG_FILE" 2>&1 || { echo "Error: XeLaTeX second pass failed"; exit 1; }

# Third run of XeLaTeX
echo "Running final XeLaTeX pass..."
xelatex -interaction=nonstopmode main.tex >> "$LOG_FILE" 2>&1 || { echo "Error: XeLaTeX final pass failed"; exit 1; }

echo "Build complete. Check $LOG_FILE for details."