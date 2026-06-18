#!/bin/bash
echo "?? Installing joblib..."
pip install joblib
echo "? joblib installed!"

echo "?? Installing all dependencies..."
pip install -r requirements.txt
echo "? All dependencies installed!"
