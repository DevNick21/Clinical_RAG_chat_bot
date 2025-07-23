@echo off
echo Starting Clinical RAG Application...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies if needed
echo Checking dependencies...
pip install -e .

REM Start the backend API server
echo Starting Flask API server...
start cmd /k "title RAG API Server && python api/app.py"

REM Wait for the API server to initialize
echo Waiting for API server to initialize...
timeout /t 5 /nobreak

REM Start the frontend
echo Starting React frontend...
cd frontend
echo Checking frontend dependencies...
call npm install
start cmd /k "title RAG Frontend && npm start"

echo.
echo Application starting! The web interface will open automatically.
echo.
echo Press any key to close this window...
pause > nul
