@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing

rem 실행 중에 .pyc 파일이 바뀌어 최신 내용 받기가 막히지 않도록 함
set PYTHONDONTWRITEBYTECODE=1

rem 최신 내용을 받은 뒤 이 파일을 새로 다시 읽어 실행한 경우
if "%~1"=="updated" goto install

rem 1) GitHub 최신 내용 받기 (git clone으로 받은 폴더일 때만)
if not exist ".git" goto no_git_folder
where git >nul 2>nul
if errorlevel 1 goto no_git_program
echo GitHub에서 최신 내용을 받는 중입니다...
git checkout -- "*.pyc" >nul 2>nul
rem 이 파일 자체가 바뀔 수 있으므로, 받은 직후 같은 줄에서 새 파일로 다시 시작
git pull --ff-only || (echo. & echo 최신 내용을 받지 못했습니다. 인터넷 연결과, 이 컴퓨터에서 직접 고친 파일이 있는지 확인하세요. & echo 지금 가지고 있는 버전으로 실행합니다.) & call "%~f0" updated & exit /b

:no_git_folder
echo 이 폴더는 git clone으로 받은 폴더가 아니라서 최신 내용 받기를 건너뜁니다.
goto install

:no_git_program
echo Git이 설치되어 있지 않아 최신 내용 받기를 건너뜁니다.
goto install

:install
rem 2) 필요한 프로그램 목록이 바뀌었으면 맞춰 설치 (이미 맞으면 바로 넘어감)
".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt
if errorlevel 1 echo 필요한 프로그램을 확인하지 못했습니다. 지금 설치된 상태로 실행합니다.

rem 3) 앱 실행
".venv\Scripts\python.exe" run.py
if errorlevel 1 pause
exit /b

:missing
echo 처음설치 파일을 먼저 실행하세요.
pause
exit /b 1
