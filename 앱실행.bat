@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" run.py
if errorlevel 1 pause
exit /b
:missing
echo 처음설치 파일을 먼저 실행하세요.
pause
exit /b 1
