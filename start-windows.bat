@echo off
rem SignaCulture AI - native Windows launcher: showcase at /, Studio at /studio/, API at /api.
cd /d "%~dp0"
set PYTHONUTF8=1
if not defined PORT set PORT=8080
rem Optional features: enabled only when their native Windows environments exist.
if exist ".venv-motion-win\Scripts\python.exe" set SIGNACULTURE_MOTION_PYTHON=%~dp0.venv-motion-win\Scripts\python.exe
if exist ".venv-pose-win\Scripts\python.exe" set SIGNACULTURE_LEXICAL_PYTHON=%~dp0.venv-pose-win\Scripts\python.exe
if not exist ".venv-win\Scripts\python.exe" (
  echo Missing .venv-win. See WINDOWS_RESTORE_STATUS.md to recreate it.
  exit /b 1
)
echo SignaCulture: http://127.0.0.1:%PORT%/   (Studio: http://127.0.0.1:%PORT%/studio/)   Ctrl+C to stop.
start "" cmd /c "timeout /t 5 >nul & start http://127.0.0.1:%PORT%/"
".venv-win\Scripts\python.exe" scripts\serve-site.py
