#!/bin/bash
# Build script for compiling the LaTeX document

# Move to the academic_paper directory
cd academic_paper

# First run of XeLaTeX
echo "Running first XeLaTeX pass..."
xelatex -interaction=nonstopmode main.tex

# Run Biber for bibliography
echo "Running Biber..."
biber main

# Second run of XeLaTeX
echo "Running second XeLaTeX pass..."
xelatex -interaction=nonstopmode main.tex

# Third run of XeLaTeX to resolve all references
echo "Running final XeLaTeX pass..."
xelatex -interaction=nonstopmode main.tex

echo "Build complete. Check for any errors above."