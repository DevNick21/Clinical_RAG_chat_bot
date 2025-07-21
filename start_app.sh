#!/bin/bash
echo "Starting Clinical RAG Application..."
echo

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
echo "Checking dependencies..."
pip install -e .

# Start the backend API server
echo "Starting Flask API server..."
(python api/app.py &) &

# Wait for the API server to initialize
echo "Waiting for API server to initialize..."
sleep 5

# Start the frontend
echo "Starting React frontend..."
cd frontend
echo "Checking frontend dependencies..."
npm install
npm start

echo
echo "Application started! The web interface should open automatically."
