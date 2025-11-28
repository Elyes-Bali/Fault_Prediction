#!/usr/bin/env bash

# 1. Install Python dependencies (in the backend folder)
echo "Installing Python dependencies..."
pip install -r backend/requirements.txt

# 2. Build the React frontend (in the frontend folder)
echo "Building React frontend..."
# Assuming package.json is in the 'frontend' directory:
npm install --prefix frontend
npm run build --prefix frontend
# This will create a 'frontend/dist' directory.

# 3. Start the combined Flask/Gunicorn server
echo "Starting Gunicorn server..."
# Gunicorn will start the server.py inside the backend directory.
# The server will run from the root of the project.
cd backend
gunicorn -w 2 --threads 4 -b 0.0.0.0:$PORT server:app 
# NOTE: Assumes your Flask instance is named 'app' inside 'server.py'