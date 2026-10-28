@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer was not found on PATH. See README.md.
  pause
  exit /b 1
)
python "%~dp0build_game.py"
set "BUILD_RESULT=%ERRORLEVEL%"
if not "%BUILD_RESULT%"=="0" echo Build failed with exit code %BUILD_RESULT%.
pause
exit /b %BUILD_RESULT%
