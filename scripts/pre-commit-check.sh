#!/bin/bash
# Pre-commit checks script for Quantifyr
# Run this before committing to ensure CI passes

echo "[INFO] Running pre-commit checks..."

# 1. Code formatting with Black
echo "[1/4] Checking code formatting with Black..."
black --check .
if [ $? -ne 0 ]; then
    echo "[FAIL] Black formatting failed! Run 'black .' to fix."
    exit 1
fi
echo "[PASS] Black formatting passed"

# 2. Linting with flake8
echo "[2/4] Linting with flake8..."
flake8 src/
if [ $? -ne 0 ]; then
    echo "[FAIL] flake8 linting failed!"
    exit 1
fi
echo "[PASS] flake8 linting passed"

# 3. Run tests
echo "[3/4] Running tests..."
pytest --maxfail=1 --disable-warnings -q
if [ $? -ne 0 ]; then
    echo "[FAIL] Tests failed!"
    exit 1
fi
echo "[PASS] Tests passed"

# 4. Check if docs build
echo "[4/4] Building documentation..."
mkdocs build --site-dir site
if [ $? -ne 0 ]; then
    echo "[FAIL] Documentation build failed!"
    exit 1
fi
echo "[PASS] Documentation build passed"

echo "[SUCCESS] All pre-commit checks passed! Safe to commit." 