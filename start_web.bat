@echo off
REM Arogya AI Web Interface Startup Script for Windows

echo 🌿 Starting Arogya AI Web Interface...
echo =====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies. Please check your internet connection.
    pause
    exit /b 1
)

REM Check if model exists
if not exist "random_forest_model.pkl" (
    echo 🔧 Model not found. Training model...
    python train_model.py
    
    if errorlevel 1 (
        echo ❌ Failed to train model. Please check the training script.
        pause
        exit /b 1
    )
)

REM Start the web application
echo 🚀 Starting web server...
echo 📱 Open your browser and go to: http://localhost:5000
echo 🛑 Press Ctrl+C to stop the server
echo.

python app.py

pause