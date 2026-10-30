@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\spectrum_inverse_design\refine_continuous_geometries.py --device cpu
if errorlevel 1 exit /b %errorlevel%
endlocal
