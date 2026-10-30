@echo off
setlocal
cd /d "%~dp0\..\.."

echo ============================================================
echo Generate the locked strict-blind benchmark
echo ============================================================
python scripts\spectrum_inverse_design\generate_locked_blind_targets.py --device auto --distance-device auto --overwrite
if errorlevel 1 goto :error

echo.
echo ============================================================
echo Run strict-blind inference and exhaustive material-sequence screening
echo ============================================================
python scripts\spectrum_inverse_design\screen_material_sequences.py --device auto --overwrite
if errorlevel 1 goto :error

echo.
echo Blind-target generation and material-sequence screening PASSED.
exit /b 0

:error
echo.
echo Pipeline stopped because a stage failed.
exit /b 1
