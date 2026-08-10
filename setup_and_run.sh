#!/bin/bash

echo "============================================================"
echo "   DO-FMBS Research Project - Setup and Launch"
echo "============================================================"
echo ""
echo "Researcher: Lydiah Moraa Machora"
echo "Reg No: I152/P/9515/23"
echo "University: Jaramogi Oginga Odinga University"
echo ""
echo "============================================================"
echo ""

# Step 1: Check Python installation
echo "[1/6] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python not found!"
    echo "Please install Python 3.10 from: https://www.python.org/downloads/"
    exit 1
fi
echo "[OK] Python found: $(python3 --version)"
echo ""

# Step 2: Create virtual environment
echo "[2/6] Setting up virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
else
    python3 -m venv venv
    echo "[OK] Virtual environment created."
fi
echo ""

# Step 3: Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "[OK] Virtual environment activated."
echo ""

# Step 4: Upgrade pip
echo "[4/6] Upgrading pip..."
pip install --upgrade pip
echo ""

# Step 5: Install dependencies
echo "[5/6] Installing required packages..."
pip install tensorflow-cpu==2.13.0
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.2
pip install numpy==1.24.3
pip install scipy==1.11.1
pip install pandas==2.0.0
pip install plotly==5.17.0
pip install streamlit==1.28.0
echo "[OK] All packages installed."
echo ""

# Step 6: Launch application
echo "============================================================"
echo "   Launching DO-FMBS Application..."
echo "============================================================"
echo ""
streamlit run app.py