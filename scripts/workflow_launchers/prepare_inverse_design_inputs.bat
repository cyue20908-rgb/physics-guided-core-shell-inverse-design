@echo off
setlocal
cd /d "%~dp0\..\.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

python scripts\verification\run_environment_preflight.py --device auto --pymiediff-device cpu
if errorlevel 1 goto :failed

python scripts\spectrum_inverse_design\prepare_inverse_design_data.py
if errorlevel 1 goto :failed

echo.
echo Inverse-design input preparation completed successfully.
pause
exit /b 0

:failed
echo.
echo Inverse-design input preparation failed. Review the traceback above.
pause
exit /b 1
