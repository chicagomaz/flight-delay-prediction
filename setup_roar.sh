#!/bin/bash
# Setup script for Penn State ROAR Collab cluster

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   Flight Delay Prediction - ROAR Collab Setup                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Load required modules
echo "📦 Loading modules..."
module purge
module load python/3.11.2

echo "✅ Modules loaded:"
module list
echo ""

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists"
    read -p "Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo "🗑️  Removed old environment"
    else
        echo "ℹ️  Using existing environment"
        source venv/bin/activate
        echo "✅ Environment activated"
        exit 0
    fi
fi

python3 -m venv venv
source venv/bin/activate

echo "✅ Virtual environment created"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo ""
echo "📦 Installing dependencies (this may take 5-10 minutes)..."
pip install -r requirements.txt

echo ""
echo "✅ Verifying installation..."
python3 << EOF
import sys
import importlib

packages = {
    'pyspark': 'PySpark',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML'
}

all_ok = True
for module, name in packages.items():
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"   ✅ {name}: {version}")
    except ImportError:
        print(f"   ❌ {name}: NOT FOUND")
        all_ok = False

if all_ok:
    print("\n✨ All packages installed successfully!")
    sys.exit(0)
else:
    print("\n⚠️  Some packages failed to install")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                 SETUP COMPLETE! ✨                             ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "🎯 Next Steps on ROAR Collab:"
    echo ""
    echo "   1. Copy your data file to the cluster:"
    echo "      scp data/raw/dot_flights_5years.csv USERNAME@submit.aci.ics.psu.edu:~/flight-delay-predictionv2/data/raw/"
    echo ""
    echo "   2. Submit training job:"
    echo "      sbatch submit_training.slurm"
    echo ""
    echo "   3. Check job status:"
    echo "      squeue -u \$USER"
    echo ""
    echo "   4. View output:"
    echo "      tail -f output/logs/slurm_JOBID.out"
    echo ""
    echo "   5. For 10% sample (faster test):"
    echo "      sbatch submit_training.slurm 0.1"
    echo ""
else
    echo ""
    echo "❌ Setup failed. Check error messages above."
    exit 1
fi
