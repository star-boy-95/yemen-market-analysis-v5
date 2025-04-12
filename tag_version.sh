#!/bin/bash
# Script to tag a new version of the Yemen Market Analysis package

VERSION="2.0.0"
MESSAGE="Major refactoring - New modular architecture"

# Add all files to staging
git add .

# Commit changes
git commit -m "Refactor: Reorganized codebase with improved architecture"

# Create an annotated tag
git tag -a "v$VERSION" -m "$MESSAGE"

echo "Created tag v$VERSION"
echo "To push this tag to the remote repository, run:"
echo "git push origin v$VERSION"
