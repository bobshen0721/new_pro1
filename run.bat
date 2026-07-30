@echo off
chcp 65001 > nul
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYANNOTE_METRICS_ENABLED=0
if not exist ".venv\Scripts\python.exe" (
  echo 找不到 .venv，請先依 README 建立環境。
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python app.py
pause
