@echo off
REM ============================================================
REM verify.bat -- run inside an activated conda env (e.g. NLP)
REM Usage:
REM   verify.bat          stages 1-4 (quick smoke, seconds)
REM   verify.bat full     also stages 5-6 (full run + pytest)
REM All lines kept short so copy-paste cannot wrap them.
REM ============================================================
setlocal
set "PYTHONUTF8=1"

cd /d "%~dp0"
echo Working directory: %CD%

where python >nul 2>nul || goto :no_python
echo Using interpreter:
python --version
echo.

echo ===== Stage 1/6: syntax compile check =====
python -m py_compile main.py
if errorlevel 1 goto :fail
python -m compileall -q src
if errorlevel 1 goto :fail
python -m compileall -q tests
if errorlevel 1 goto :fail
echo OK: all .py files compile
echo.

echo ===== Stage 2/6: dependency check =====
for %%M in (numpy pandas sklearn scipy yaml) do (
    python -c "import %%M" 2>nul
    if errorlevel 1 (
        echo Missing dependency: %%M
        echo If yaml is missing run:  pip install pyyaml pytest
        goto :fail
    )
)
python -c "import sklearn; print('OK: sklearn', sklearn.__version__)"
if errorlevel 1 goto :fail
echo.

echo ===== Stage 3/6: data file check =====
if not exist "..\secom.data" (
    echo DATA FILE MISSING: ..\secom.data
    echo Copy the WHOLE Secom folder, not only this subfolder.
    goto :fail
)
if not exist "..\secom_labels.data" (
    echo DATA FILE MISSING: ..\secom_labels.data
    goto :fail
)
if not exist "baseline_metrics.json" (
    echo [ERROR] baseline_metrics.json is missing
    goto :fail
)
echo OK: data and baseline files present
echo.

echo ===== Stage 4/6: quick smoke run (no RFE, seconds) =====
python main.py --quick
if errorlevel 1 goto :fail
echo OK: quick smoke passed
echo.

if /i not "%~1"=="full" (
    echo Smoke passed. For full verification run: verify.bat full
    exit /b 0
)

echo ===== Stage 5/6: full pipeline + baseline check (1-3 min) =====
python main.py
if errorlevel 1 goto :fail
echo OK: full pipeline passed, metrics within tolerance
echo.

echo ===== Stage 6/6: pytest regression tests =====
python -c "import pytest" 2>nul
if errorlevel 1 (
    echo pytest missing. Run: pip install pytest
    goto :fail
)
python -m pytest tests -v
if errorlevel 1 goto :fail
echo.
echo ALL CHECKS PASSED
exit /b 0

:no_python
echo [ERROR] python not found. Activate your env first:
echo   conda activate NLP
exit /b 1

:fail
echo.
echo [VERIFICATION FAILED] See output above.
exit /b 1
