#!/bin/bash
# Arogya AI Web Interface Startup Script

echo "🌿 Starting Arogya AI Web Interface..."
echo "====================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Please check your internet connection."
    exit 1
fi

# Check if model exists
if [ ! -f "random_forest_model.pkl" ]; then
    echo "🔧 Model not found. Training model..."
    python train_model.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to train model. Please check the training script."
        exit 1
    fi
fi

# Start the web application
echo "🚀 Starting web server..."
echo "📱 Open your browser and go to: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python app.py