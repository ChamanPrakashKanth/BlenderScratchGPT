@echo off
title Sparse-AST Model & Top-K MoE Console
cd /d "%~dp0"

echo =====================================================================
echo           Sparse-AST AI Model & Top-K MoE Ensemble Console
echo =====================================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH!
    echo Please make sure Python 3.8+ is installed.
    echo.
    pause
    exit /b 1
)

python chat.py

if errorlevel 1 (
    echo.
    echo [NOTE] The process exited with code %errorlevel%.
    pause
)
