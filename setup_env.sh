#!/bin/bash
# Setup Python virtual environment for Flight Delay Prediction

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Flight Delay Prediction - Environment Setup              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check Java
echo "📋 Step 1/4: Checking Java installation..."
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n 1)
    echo "   ✅ Java found: $JAVA_VERSION"
else
    echo "   ❌ Java not found!"
    echo ""
    echo "   Installing OpenJDK 11 (required for PySpark)..."
    sudo apt update
    sudo apt install -y openjdk-11-jdk
    echo "   ✅ Java installed"
fi
echo ""

# Step 2: Create virtual environment
echo "🐍 Step 2/4: Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "   ⚠️  Virtual environment already exists"
    read -p "   Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo "   🗑️  Removed old environment"
    else
        echo "   ℹ️  Using existing environment"
    fi
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi
echo ""

# Step 3: Activate and install dependencies
echo "📦 Step 3/4: Installing dependencies (this may take 5-10 minutes)..."
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install from requirements.txt
echo "   Installing packages from requirements.txt..."
pip install -r requirements.txt

echo "   ✅ All dependencies installed"
echo ""

# Step 4: Verify installation
echo "✅ Step 4/4: Verifying installation..."

# Test critical imports
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
    echo "║                    SETUP COMPLETE! ✨                          ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "🎯 Next Steps:"
    echo ""
    echo "   1. Activate environment:"
    echo "      source venv/bin/activate"
    echo ""
    echo "   2. Test with 1% sample (~5 min):"
    echo "      ./scripts/auto_train_after_download.sh 0.01"
    echo ""
    echo "   3. Run full training (~30-60 min):"
    echo "      ./scripts/auto_train_after_download.sh"
    echo ""
    echo "   4. Deactivate when done:"
    echo "      deactivate"
    echo ""
else
    echo ""
    echo "❌ Setup failed. Check error messages above."
    exit 1
fi
