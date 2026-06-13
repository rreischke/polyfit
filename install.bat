@echo off
:: ================================================================
::  polyfit -- Windows Installer
::
::  Double-click this file to install polyfit.
::  A Command Prompt window will open and guide you through
::  the process.
:: ================================================================

chcp 65001 >nul 2>&1
title polyfit Installer

echo ========================================
echo          polyfit -- Installer
echo ========================================
echo.
echo This installer will:
echo   1. Install the polyfit Python package
echo   2. Create a polyfit.bat launcher
echo   3. Create a polyfit shortcut on your Desktop
echo.
echo ----------------------------------------
echo.

:: ----------------------------------------------------------------
:: Find Python 3
:: ----------------------------------------------------------------
echo Checking for Python 3...

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python was not found on this system.
    echo.
    echo Please install Python 3 from:  https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo Then double-click this installer again.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo   Found: %%v
echo.

:: ----------------------------------------------------------------
:: Run the Python installer
:: ----------------------------------------------------------------
echo Starting installation...
echo.

python "%~dp0create_app.py"
set EXIT_CODE=%ERRORLEVEL%
echo.

if %EXIT_CODE% equ 0 (
    echo Installation complete!
    echo.
    echo   ^> Double-click the polyfit shortcut on your Desktop to launch.
    echo   ^> Or run polyfit.bat directly from this folder.
) else (
    echo Installation failed ^(exit code: %EXIT_CODE%^).
    echo.
    echo Please check the error messages above.
    echo If the problem persists, try running manually:
    echo.
    echo   cd /d "%~dp0"
    echo   python create_app.py
)

echo.
pause
