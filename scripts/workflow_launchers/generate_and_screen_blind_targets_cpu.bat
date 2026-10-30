@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\spectrum_inverse_design\generate_locked_blind_targets.py --device cpu --distance-device cpu
if errorlevel 1 exit /b %errorlevel%
python scripts\spectrum_inverse_design\screen_material_sequences.py --device cpu
endlocal
