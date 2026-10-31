#!/usr/bin/env bash
# ==============================================================================
# Setup script for M4 MacBook Air (Cosmology Research & CosmoBolognaLib)
# ==============================================================================
set -e

echo "🚀 Starting M4 MacBook Air research environment setup..."

# 1. Homebrew check & installation
if ! command -v brew &>/dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
else
    echo "✅ Homebrew already installed."
fi

# Ensure brew is in PATH for this script session
eval "$(/opt/homebrew/bin/brew shellenv)"

# 2. Install required compilers and C++ scientific libraries
echo "📦 Installing GCC, GSL, FFTW, Boost, CMake..."
brew install gcc@15 gsl fftw boost cmake cfitsio libomp

# 3. Install Python 3.12 and scientific stack
echo "🐍 Setting up Python 3.12 and scientific libraries..."
brew install python@3.12
/opt/homebrew/bin/python3.12 -m pip install --upgrade pip
/opt/homebrew/bin/python3.12 -m pip install numpy scipy pandas emcee matplotlib corner getdist

# 4. Rebuild CosmoBolognaLib with GCC-15 on M4
CBL_DIR="$HOME/projects/CosmoBolognaLib"
if [ -d "$CBL_DIR" ]; then
    echo "⚙️ Rebuilding CosmoBolognaLib in $CBL_DIR..."
    cd "$CBL_DIR"
    make clean 2>/dev/null || true
    make CUBA -B
    make -j8
    make python -B PY=/opt/homebrew/bin/python3.12
    echo "✅ CosmoBolognaLib compilation complete!"
else
    echo "⚠️ Warning: $CBL_DIR not found. Make sure to copy CosmoBolognaLib to ~/projects/ first."
fi

# 5. Fix permissions for SSH keys
if [ -d "$HOME/.ssh" ]; then
    chmod 700 "$HOME/.ssh"
    chmod 600 "$HOME/.ssh"/id_* 2>/dev/null || true
    chmod 644 "$HOME/.ssh"/*.pub 2>/dev/null || true
    chmod 644 "$HOME/.ssh"/config 2>/dev/null || true
    echo "🔒 SSH permissions secured."
fi

# 6. Verify CBL Python import
echo "🧪 Verifying CosmoBolognaLib python import..."
export CBL_HOME="$CBL_DIR"
export PYTHONPATH="$CBL_HOME/Python:$PYTHONPATH"
export DYLD_LIBRARY_PATH="$CBL_DIR:/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
/opt/homebrew/bin/python3.12 -c "import CosmoBolognaLib as cbl; print('🎉 CBL loaded successfully on M4! Version:', cbl.__file__)"

echo ""
echo "=================================================================="
echo "🎉 Setup complete! All libraries, compiler shims, and CBL are ready."
echo "You can now launch Antigravity on your M4 and pick up right where you left off!"
echo "=================================================================="
