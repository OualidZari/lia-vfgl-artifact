#!/bin/bash

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found. Please install Python 3 and try again."
    exit 1
fi

# Try to install pip, ensurepip, and parallel
echo "Attempting to install pip, ensurepip, and parallel..."
sudo apt update
sudo apt install -y python3-pip python3-venv parallel

# Check if pip is now available
if ! command -v pip3 &> /dev/null; then
    echo "Failed to install pip. Please install it manually and run this script again."
    exit 1
fi

# Check if parallel is now available
if ! command -v parallel &> /dev/null; then
    echo "Failed to install parallel. Please install it manually and run this script again."
    exit 1
fi

# Create a virtual environment
python3 -m venv lia_vfgl_env

# Check if virtual environment was created successfully
if [ ! -d "lia_vfgl_env" ]; then
    echo "Failed to create virtual environment. Please check your Python installation and try again."
    exit 1
fi

# Activate the virtual environment
source lia_vfgl_env/bin/activate

# Upgrade pip in the virtual environment
pip install --upgrade pip

# Install torch first
pip install torch==1.13.0

# Install requirements
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping package installation."
fi

# Download datasets
echo "Downloading datasets..."
python download_datasets.py

echo "Environment setup complete. You can activate it with 'source lia_vfgl_env/bin/activate'"

# Deactivate the environment
deactivate