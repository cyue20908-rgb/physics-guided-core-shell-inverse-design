@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\spectrum_inverse_design\train_spectrum_conditioned_inverse_network.py --device auto
if errorlevel 1 exit /b %errorlevel%
echo.
echo Spectrum-conditioned inverse-network training completed successfully.
endlocal
