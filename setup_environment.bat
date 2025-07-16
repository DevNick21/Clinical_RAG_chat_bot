@echo off
REM ============================================================================
REM Clinical RAG System - Environment Setup Script (Windows)
REM ============================================================================
REM Automated setup script for the MIMIC-IV Clinical RAG system on Windows
REM This script creates a Python virtual environment and installs dependencies
REM Last updated: July 2025

setlocal enabledelayedexpansion

echo 🏥 Clinical RAG System - Environment Setup (Windows)
echo ===================================================

REM Check if Python is installed
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo [ERROR] Please install Python 3.11 or later first
    echo [ERROR] Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [SUCCESS] Python found
python --version

REM Check Python version (should be 3.11+)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Python version: %PYTHON_VERSION%

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo [ERROR] Requirements file 'requirements.txt' not found
    echo [ERROR] Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check if virtual environment already exists
if exist "venv\" (
    echo [WARNING] Virtual environment 'venv' already exists
    set /p "recreate=Do you want to recreate it? (y/n): "
    if /i "!recreate!"=="y" (
        echo [INFO] Removing existing virtual environment...
        rmdir /s /q venv
        echo [INFO] Creating new virtual environment...
        python -m venv venv
    ) else (
        echo [INFO] Using existing virtual environment
    )
) else (
    echo [INFO] Creating new virtual environment 'venv'...
    python -m venv venv
)

echo [SUCCESS] Virtual environment ready!

REM Activate environment and install dependencies
echo [INFO] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat

REM Upgrade pip to latest version
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo [INFO] Installing project dependencies...
pip install -r requirements.txt

echo [SUCCESS] Dependencies installed!

REM Verify key packages
echo [INFO] Verifying package installation...
python -c "
import sys
print(f'Python version: {sys.version}')

packages = [
    ('langchain', 'LangChain'),
    ('faiss', 'FAISS'),
    ('sentence_transformers', 'Sentence Transformers'),
    ('pandas', 'Pandas'),
    ('torch', 'PyTorch')
]

for module, name in packages:
    try:
        __import__(module)
        print(f'✓ {name} imported successfully')
    except ImportError as e:
        print(f'✗ {name} import failed: {e}')
"

echo [SUCCESS] Environment verification completed!
echo.
echo 🎉 Setup Complete!
echo ==================
echo Your clinical RAG virtual environment is ready to use.
echo.
echo To activate the environment manually:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate the environment:
echo   deactivate
echo.
echo To start Jupyter notebook:
echo   venv\Scripts\activate.bat
echo   jupyter notebook
echo.
echo To run the main RAG pipeline:
echo   venv\Scripts\activate.bat
echo   python RAG_chat_pipeline\main.py
echo.

REM Optional: Create batch file for quick activation
set /p "shortcut=Would you like to create a 'activate-rag.bat' shortcut to activate this environment? (y/n): "
if /i "%shortcut%"=="y" (
    echo @echo off > activate-rag.bat
    echo call venv\Scripts\activate.bat >> activate-rag.bat
    echo cmd /k >> activate-rag.bat
    echo [SUCCESS] Created 'activate-rag.bat' shortcut
    echo [INFO] Double-click 'activate-rag.bat' to activate the environment
)

echo [SUCCESS] All done! Happy coding! 🚀
pause
