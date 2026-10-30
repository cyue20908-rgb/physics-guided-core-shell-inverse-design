@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\spectrum_inverse_design\freeze_blind_designs.py
endlocal
