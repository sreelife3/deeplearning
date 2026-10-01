@echo off
REM ===================================================================
REM  HAR study runner  -  Windows
REM
REM    run_study.bat          setup, self-test, smoke run, full sweep, report
REM    run_study.bat quick    setup, self-test, smoke run only  (~3 minutes)
REM    run_study.bat report   regenerate the README results only
REM
REM  For a log file:  run_study.bat > run_log.txt 2>&1
REM ===================================================================
setlocal
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set MODE=%1
if "%MODE%"=="" set MODE=full

echo.
echo ===================================================================
echo  HAR study  -  mode: %MODE%
echo  Working directory: %CD%
echo ===================================================================

REM ---------- 0. find a Python ---------------------------------------
set PY=
python --version >nul 2>&1 && set PY=python
if not defined PY ( py -3 --version >nul 2>&1 && set PY=py -3 )
if not defined PY (
  echo [X] No Python found on PATH.
  echo     Install Python 3.10 or newer from python.org and tick
  echo     "Add python.exe to PATH" during setup, then re-run this script.
  goto :fail
)
echo [1/6] Python: & %PY% --version

REM ---------- 1. virtual environment ---------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo [2/6] Creating virtual environment .venv ...
  %PY% -m venv .venv || goto :fail
) else (
  echo [2/6] Virtual environment .venv already exists - reusing it.
)
call ".venv\Scripts\activate.bat" || goto :fail

REM ---------- 2. dependencies ----------------------------------------
if exist ".venv\.deps_ok" (
  echo [3/6] Dependencies already installed - skipping.
) else (
  echo [3/6] Installing dependencies. First run downloads PyTorch, ~200 MB.
  python -m pip install --upgrade pip --quiet || goto :fail
  python -m pip install -r requirements.txt || goto :fail
  echo ok> ".venv\.deps_ok"
)

REM ---------- 3. self-test (no dataset needed) ------------------------
echo.
echo [4/6] Self-test - shapes, quantisation, the transpose guard. ~30 seconds.
echo -------------------------------------------------------------------
python run_all.py --selftest || goto :fail
if "%MODE%"=="report" goto :report

REM ---------- 4. smoke run (proves the download and the split) --------
echo.
echo [5/6] Smoke run - one model, one seed, 3 epochs.
echo       This is the first time the UCI dataset is downloaded and split.
echo       Watch for: subject lists with no overlap, train mean ~0 std ~1,
echo       and a test mean near zero but NOT exactly zero.
echo -------------------------------------------------------------------
python run_all.py --models gru --seeds 1 --epochs 3 --no-mlflow || goto :fail

if /i "%MODE%"=="quick" (
  echo.
  echo ===================================================================
  echo  Quick mode finished. The pipeline works end to end.
  echo  Run  run_study.bat  with no argument for the full study.
  echo ===================================================================
  goto :done
)

REM ---------- 5. the full study --------------------------------------
echo.
echo [6/6] Full study - 4 models x 3 seeds. Expect 45-90 minutes.
echo       Progress prints per epoch with a running time estimate.
echo       Safe to leave unattended; safe to Ctrl-C and re-run.
echo -------------------------------------------------------------------
python run_all.py || goto :fail

REM ---------- 6. fill the README -------------------------------------
:report
echo.
echo Regenerating the README results section from results\summary.csv ...
python make_report.py || goto :fail

:done
echo.
echo ===================================================================
echo  DONE
echo -------------------------------------------------------------------
echo  results\raw.csv          every run, one row per model/seed/precision
echo  results\summary.csv      aggregated across seeds
echo  results\environment.json the machine these timings mean something on
echo  figures\pareto.png       accuracy against latency
echo  figures\confusion.png    where the errors actually are
echo  README.md                results section now populated
echo ===================================================================
call deactivate >nul 2>&1
exit /b 0

:fail
echo.
echo ===================================================================
echo  STOPPED - the stage above failed. Nothing further was run.
echo.
echo  If the dataset download failed: fetch
echo    https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
echo  in a browser, unzip it into a folder named  data  beside this script,
echo  then re-run. The loader handles the nested zip the archive ships.
echo ===================================================================
call deactivate >nul 2>&1
exit /b 1
