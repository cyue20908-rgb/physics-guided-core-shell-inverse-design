@echo off
setlocal
cd /d "%~dp0\..\.."
python scripts\spectrum_inverse_design\search_integer_geometries.py --device cpu
if errorlevel 1 (
  echo.
  echo Integer-geometry search FAILED.
  exit /b 1
)
echo.
echo Integer-geometry search completed successfully.
endlocal
