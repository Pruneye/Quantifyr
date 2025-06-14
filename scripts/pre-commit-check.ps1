#!/usr/bin/env pwsh
# Pre-commit checks script for Quantifyr
# Run this before committing to ensure CI passes

Write-Host "[INFO] Running pre-commit checks..." -ForegroundColor Cyan

# 1. Code formatting with Black
Write-Host "`n[1/4] Checking code formatting with Black..." -ForegroundColor Yellow
black .
black --check .
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Black formatting failed! Run 'black .' to fix." -ForegroundColor Red
    exit 1
}
Write-Host "[PASS] Black formatting passed" -ForegroundColor Green

# 2. Linting with flake8
Write-Host "`n[2/4] Linting with flake8..." -ForegroundColor Yellow
flake8 src/
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] flake8 linting failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[PASS] flake8 linting passed" -ForegroundColor Green

# 3. Run tests
Write-Host "`n[3/4] Running tests..." -ForegroundColor Yellow
pytest --maxfail=1 --disable-warnings -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Tests failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[PASS] Tests passed" -ForegroundColor Green

# 4. Check if docs build
Write-Host "`n[4/4] Building documentation..." -ForegroundColor Yellow
$env:JUPYTER_PLATFORM_DIRS = "1"
mkdocs build --site-dir site
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Documentation build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[PASS] Documentation build passed" -ForegroundColor Green

Write-Host "`n[SUCCESS] All pre-commit checks passed! Safe to commit." -ForegroundColor Green 