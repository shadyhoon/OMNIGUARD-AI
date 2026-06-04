@echo off
REM ============================================================
REM OmniGuard AI - Virtual Environment Setup Script (Windows)
REM ============================================================

echo.
echo ==========================================
echo  OmniGuard AI - Environment Setup
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Python found.
echo.

REM Create virtual environment
if not exist ".venv\" (
    echo [2/5] Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/5] Virtual environment already exists, skipping creation.
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment ...
call .venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [4/5] Upgrading pip ...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [5/5] Installing dependencies from requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo  Setup complete!
echo ==========================================
echo.
echo To run the dashboard:
echo   .venv\Scripts\activate
echo   streamlit run app.py
echo.
pause
