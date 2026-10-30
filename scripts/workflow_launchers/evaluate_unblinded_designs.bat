@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\spectrum_inverse_design\evaluate_unblinded_designs.py
if errorlevel 1 (
  echo.
  echo Post-unblinding evaluation failed. Review the traceback above.
  exit /b 1
)
echo.
echo Post-unblinding evaluation completed successfully.
endlocal
