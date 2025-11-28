#!/usr/bin/env bash

# This script is now designed to be run from the 'backend' directory (Render Root Directory)

# 1. Install Python dependencies
echo "Installing Python dependencies..."
# requirements.txt is now local to the current directory (backend/)
pip install -r requirements.txt

# 2. Build the React frontend
echo "Building React frontend..."
# The frontend folder is one level up and parallel to backend/
npm install --prefix ../frontend
npm run build --prefix ../frontend

# 3. Move the compiled frontend code to the correct location
# The build created ../frontend/dist. Flask needs it at ./dist (inside backend/).
echo "Moving React assets from ../frontend/dist to ./dist..."
mv ../frontend/dist ./dist

echo "Build process completed successfully."