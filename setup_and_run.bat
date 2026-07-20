@echo off
setlocal EnableExtensions
title DO-FMBS Setup and Launch
color 0A

cd /d "%~dp0"

echo ============================================================
echo    DO-FMBS Research Project - Setup and Launch
echo ============================================================
echo.

echo [1/6] Python check
set "PYTHON_CMD="
call :FindCompatiblePython

if "%PYTHON_CMD%"=="" (
    echo No TensorFlow-compatible Python found. Attempting to install Python 3.11...
    call :InstallPython311
    set "PYTHON_CMD="
    call :FindCompatiblePython
)

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Could not install or find Python 3.9, 3.10, or 3.11.
    echo This project uses tensorflow-cpu==2.13.0, which does not support Python 3.14.
    echo Install Python 3.11 from https://www.python.org/downloads/ and rerun this file.
    pause
    exit /b 1
)

%PYTHON_CMD% --version
echo.

echo [2/6] Virtual environment check
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 11) else 1)" >nul 2>&1
    if errorlevel 1 (
        echo Existing virtual environment is broken or uses a TensorFlow-incompatible Python. Recreating it...
        rmdir /s /q "venv"
        %PYTHON_CMD% -m venv venv
        if errorlevel 1 (
            echo [ERROR] Could not recreate virtual environment.
            pause
            exit /b 1
        )
    ) else (
        echo [OK] Existing virtual environment found.
    )
) else (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Could not create virtual environment.
        pause
        exit /b 1
    )
)
echo.

echo [3/6] Activating virtual environment
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Could not activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment active.
echo.

echo [4/6] Upgrading pip
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    pause
    exit /b 1
)
echo.

echo [5/6] Installing requirements
if exist "requirements_app.txt" (
    python -m pip install -r requirements_app.txt
) else (
    python -m pip install tensorflow-cpu==2.13.0 scikit-learn==1.3.0 matplotlib==3.7.2 numpy==1.24.3 scipy==1.11.1 pandas==2.0.0 plotly==5.17.0 streamlit==1.28.0 opencv-python==4.8.1.78 pillow==10.0.0
)
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)
echo.

echo [6/6] Verifying application dependencies
python -c "import streamlit, tensorflow, sklearn, pandas, plotly; print('Dependencies OK')"
if errorlevel 1 (
    echo [ERROR] Verification failed.
    pause
    exit /b 1
)
echo.

if exist "do_fmbs_data\face_data.npz" (
    echo [OK] Existing project data found.
) else (
    echo [INFO] Fresh project data folder detected. Run Phase 1 inside the app.
)
echo.

echo ============================================================
echo    Launching DO-FMBS at http://localhost:8501
echo ============================================================
echo Press Ctrl+C in this window to stop the server.
echo.

start "" cmd /c "timeout /t 5 >nul && start http://localhost:8501"
python -m streamlit run app.py --server.headless false --server.port 8501

echo.
echo Streamlit stopped.
pause
exit /b 0

:FindCompatiblePython
if defined PYTHON_CMD exit /b 0

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 11) else 1)" >nul 2>&1 && set "PYTHON_CMD=python"
)
if defined PYTHON_CMD exit /b 0

where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>&1 && set "PYTHON_CMD=py -3.11"
)
if defined PYTHON_CMD exit /b 0

where py >nul 2>&1
if not errorlevel 1 (
    py -3.10 -c "import sys" >nul 2>&1 && set "PYTHON_CMD=py -3.10"
)
if defined PYTHON_CMD exit /b 0

where py >nul 2>&1
if not errorlevel 1 (
    py -3.9 -c "import sys" >nul 2>&1 && set "PYTHON_CMD=py -3.9"
)
if defined PYTHON_CMD exit /b 0

if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    "%LocalAppData%\Programs\Python\Python311\python.exe" -c "import sys" >nul 2>&1 && set "PYTHON_CMD="%LocalAppData%\Programs\Python\Python311\python.exe""
)
if defined PYTHON_CMD exit /b 0

if exist "C:\Python311\python.exe" (
    "C:\Python311\python.exe" -c "import sys" >nul 2>&1 && set "PYTHON_CMD="C:\Python311\python.exe""
)
exit /b 0

:InstallPython311
where py >nul 2>&1
if not errorlevel 1 (
    echo Trying Python Launcher installer: py install 3.11
    py install 3.11
    py -3.11 -c "import sys" >nul 2>&1
    if not errorlevel 1 exit /b 0
)

where winget >nul 2>&1
if not errorlevel 1 (
    echo Trying Windows Package Manager installer: winget install Python 3.11
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    if exist "%LocalAppData%\Programs\Python\Python311\python.exe" exit /b 0
)

echo Automatic Python 3.11 installation was not available on this machine.
exit /b 1
