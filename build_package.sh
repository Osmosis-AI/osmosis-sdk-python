#!/bin/bash
# Script to build and upload the package to PyPI

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Install build dependencies if not already installed
pip install --upgrade build twine

# Build the package
python -m build

# Check the package
twine check dist/*

echo ""
echo "To upload to TestPyPI, run:"
echo "twine upload --repository testpypi dist/*"
echo ""
echo "To upload to PyPI, run:"
echo "twine upload dist/*" 