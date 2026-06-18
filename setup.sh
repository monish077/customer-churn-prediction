#!/bin/bash

# setup.sh - Installation script for Streamlit Cloud

echo "🔧 Setting up environment..."

# Install system dependencies
sudo apt-get update
sudo apt-get install -y zlib1g-dev libjpeg-dev libpng-dev

# Install Python packages
pip install -r requirements.txt

echo "✅ Setup complete!"