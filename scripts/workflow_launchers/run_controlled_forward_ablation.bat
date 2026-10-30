@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\forward_model_training\run_controlled_ablation.py --device auto
if errorlevel 1 exit /b %errorlevel%
echo.
echo Controlled forward-model ablation completed successfully.
endlocal
