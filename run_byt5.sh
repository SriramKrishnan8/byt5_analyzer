#!/bin/bash
set -e

# ==========================================
# 1. Environment Setup
# ==========================================
# Uncomment and adjust the path to your virtual environment if needed
# source .venv/bin/activate

# ==========================================
# 2. Automated Test Suite
# ==========================================
if [ "$1" = "--test" ]; then
    echo "Triggering the ByT5 Automated Test Suite..."
    # Run the unittest script
    python3 tests/test_byt5.py
    exit 0
fi

# ==========================================
# 3. Standard Execution
# ==========================================
# Forward all arguments passed to the shell script directly to cli.py
python3 cli.py "$@"