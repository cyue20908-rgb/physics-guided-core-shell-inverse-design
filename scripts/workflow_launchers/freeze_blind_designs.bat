@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\spectrum_inverse_design\freeze_blind_designs.py
if errorlevel 1 (
  echo Pre-unblinding integrity freeze failed.
  exit /b 1
)
echo Pre-unblinding integrity freeze completed successfully.
endlocal
