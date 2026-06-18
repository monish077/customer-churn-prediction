#!/bin/bash
echo "?? Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y zlib1g-dev libjpeg-dev libpng-dev
echo "? System dependencies installed!"
