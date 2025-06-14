#!/bin/bash
# Pre-commit checks script for Quantifyr
# Run this before committing to ensure CI passes

echo "[INFO] Running pre-commit checks..."

# 1. Auto-fix imports and unused code
echo "[1/6] Auto-fixing imports and unused code..."
autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive src/
echo "[PASS] Auto-fix imports completed"

# 2. Auto-fix code style issues
echo "[2/6] Auto-fixing code style with autopep8..."
autopep8 --in-place --aggressive --aggressive --recursive src/
echo "[PASS] Auto-fix code style completed"

# 3. Code formatting with Black
echo "[3/6] Applying Black formatting..."
black .
black --check .
if [ $? -ne 0 ]; then
    echo "[FAIL] Black formatting failed! Run 'black .' to fix."
    exit 1
fi
echo "[PASS] Black formatting passed"

# 4. Final linting check with flake8
echo "[4/6] Final linting check with flake8..."
flake8 src/ --ignore=E501
if [ $? -ne 0 ]; then
    echo "[FAIL] flake8 linting failed after auto-fixes!"
    echo "[INFO] Some issues may need manual fixing"
    exit 1
fi
echo "[PASS] flake8 linting passed"

# 5. Run tests
echo "[5/6] Running tests..."
pytest --maxfail=1 -q
if [ $? -ne 0 ]; then
    echo "[FAIL] Tests failed!"
    exit 1
fi
echo "[PASS] Tests passed"

# 6. Check if docs build
echo "[6/6] Building documentation..."
export JUPYTER_PLATFORM_DIRS=1
mkdocs build --site-dir site
if [ $? -ne 0 ]; then
    echo "[FAIL] Documentation build failed!"
    exit 1
fi
echo "[PASS] Documentation build passed"

echo "[SUCCESS] All pre-commit checks passed! Safe to commit." 