@echo off
title 分贝记录仪
cd /d "%~dp0"
echo 正在启动分贝记录仪...
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe decibel_meter.py
    if errorlevel 1 pause
) else (
    echo 未找到虚拟环境，请先执行:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install sounddevice numpy matplotlib
    pause
)
