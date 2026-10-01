#!/usr/bin/env bash
# ===================================================================
#  HAR study runner  -  macOS / Linux / WSL
#
#    ./run_study.sh          setup, self-test, smoke run, full sweep, report
#    ./run_study.sh quick    setup, self-test, smoke run only  (~3 minutes)
#    ./run_study.sh report   regenerate the README results only
#
#  For a log file:  ./run_study.sh 2>&1 | tee run_log.txt
# ===================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
MODE="${1:-full}"

trap 'echo; echo "==================================================================="; \
      echo " STOPPED - the stage above failed. Nothing further was run."; \
      echo; echo " If the dataset download failed, fetch"; \
      echo "   https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"; \
      echo " and unzip it into a folder named  data  beside this script, then re-run."; \
      echo "==================================================================="' ERR

echo
echo "==================================================================="
echo " HAR study  -  mode: $MODE"
echo " Working directory: $PWD"
echo "==================================================================="

PY=python3
command -v "$PY" >/dev/null || PY=python
echo "[1/6] Python: $($PY --version)"

if [ ! -x ".venv/bin/python" ]; then
  echo "[2/6] Creating virtual environment .venv ..."
  "$PY" -m venv .venv
else
  echo "[2/6] Virtual environment .venv already exists - reusing it."
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [ -f ".venv/.deps_ok" ]; then
  echo "[3/6] Dependencies already installed - skipping."
else
  echo "[3/6] Installing dependencies. First run downloads PyTorch, ~200 MB."
  python -m pip install --upgrade pip --quiet
  python -m pip install -r requirements.txt
  touch .venv/.deps_ok
fi

echo
echo "[4/6] Self-test - shapes, quantisation, the transpose guard. ~30 seconds."
echo "-------------------------------------------------------------------"
python run_all.py --selftest

if [ "$MODE" = "report" ]; then
  python make_report.py
  exit 0
fi

echo
echo "[5/6] Smoke run - one model, one seed, 3 epochs."
echo "      This is the first time the UCI dataset is downloaded and split."
echo "      Watch for: subject lists with no overlap, train mean ~0 std ~1,"
echo "      and a test mean near zero but NOT exactly zero."
echo "-------------------------------------------------------------------"
python run_all.py --models gru --seeds 1 --epochs 3 --no-mlflow

if [ "$MODE" = "quick" ]; then
  echo
  echo "==================================================================="
  echo " Quick mode finished. The pipeline works end to end."
  echo " Run ./run_study.sh with no argument for the full study."
  echo "==================================================================="
  exit 0
fi

echo
echo "[6/6] Full study - 4 models x 3 seeds. Expect 45-90 minutes."
echo "      Progress prints per epoch with a running time estimate."
echo "      Safe to leave unattended; safe to Ctrl-C and re-run."
echo "-------------------------------------------------------------------"
python run_all.py

echo
echo "Regenerating the README results section from results/summary.csv ..."
python make_report.py

echo
echo "==================================================================="
echo " DONE"
echo "-------------------------------------------------------------------"
echo " results/raw.csv          every run, one row per model/seed/precision"
echo " results/summary.csv      aggregated across seeds"
echo " results/environment.json the machine these timings mean something on"
echo " figures/pareto.png       accuracy against latency"
echo " figures/confusion.png    where the errors actually are"
echo " README.md                results section now populated"
echo "==================================================================="
