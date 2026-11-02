@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3.12 -m venv ".venv"
if errorlevel 1 goto use_python
goto install
:use_python
python -c "import sys;sys.exit(0 if sys.version_info[:2]==(3,12) else 1)"
if errorlevel 1 goto missing_python
python -m venv ".venv"
if errorlevel 1 goto failure
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failure
".venv\Scripts\python.exe" run.py --check
if errorlevel 1 goto failure
echo.
echo 설치가 끝났습니다. 이제 앱실행 파일을 더블클릭하세요.
pause
exit /b 0
:missing_python
echo 파이썬 3.12를 먼저 설치한 뒤 다시 실행하세요.
goto failure
:failure
echo 설치를 마치지 못했습니다. 위 오류 내용을 확인하세요.
pause
exit /b 1
