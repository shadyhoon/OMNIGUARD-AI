#!/bin/bash
# ============================================================
# OmniGuard AI - Virtual Environment Setup Script (Linux/macOS)
# ============================================================

set -e

echo
echo "=========================================="
echo " OmniGuard AI - Environment Setup"
echo "=========================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH."
    exit 1
fi

echo "[1/5] Python found: $(python3 --version)"
echo

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "[2/5] Creating virtual environment in .venv ..."
    python3 -m venv .venv
else
    echo "[2/5] Virtual environment already exists, skipping creation."
fi
echo

# Activate virtual environment
echo "[3/5] Activating virtual environment ..."
# shellcheck disable=SC1091
source .venv/bin/activate
echo

# Upgrade pip
echo "[4/5] Upgrading pip ..."
python -m pip install --upgrade pip
echo

# Install dependencies
echo "[5/5] Installing dependencies from requirements.txt ..."
pip install -r requirements.txt

echo
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
echo
echo "To run the dashboard:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py"
echo
