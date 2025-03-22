@echo off
REM Script to build and upload the package to PyPI

REM Clean previous builds
rmdir /s /q dist build osmosis_ai.egg-info

REM Install build dependencies if not already installed
pip install --upgrade build twine

REM Build the package
python -m build

REM Check the package
twine check dist/*

echo.
echo To upload to TestPyPI, run:
echo twine upload --repository testpypi dist/*
echo.
echo To upload to PyPI, run:
echo twine upload dist/* 