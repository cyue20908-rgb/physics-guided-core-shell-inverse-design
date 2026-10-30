@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\robust_design\run_robust_design_pipeline.py --device cpu
endlocal
