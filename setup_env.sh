#!/bin/bash
# Script to set up a Python virtual environment for osmosis-ai

set -e  # Exit on error

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e .
pip install -r requirements.txt

# Setup environment variables
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.sample .env
    echo "Please edit .env file to add your API keys"
fi

echo ""
echo "Environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "Remember to edit your .env file to add your API keys." 