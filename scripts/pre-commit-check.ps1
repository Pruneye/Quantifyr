#!/usr/bin/env pwsh
# Pre-commit checks script for Quantifyr
# Run this before committing to ensure CI passes

Write-Host "[INFO] Running pre-commit checks..." -ForegroundColor Cyan

# 1. Auto-fix imports and unused code
Write-Host "`n[1/6] Auto-fixing imports and unused code..." -ForegroundColor Yellow
autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive src/
Write-Host "[PASS] Auto-fix imports completed" -ForegroundColor Green

# 2. Auto-fix code style issues
Write-Host "`n[2/6] Auto-fixing code style with autopep8..." -ForegroundColor Yellow
autopep8 --in-place --aggressive --aggressive --recursive src/
Write-Host "[PASS] Auto-fix code style completed" -ForegroundColor Green

# 3. Code formatting with Black
Write-Host "`n[3/6] Applying Black formatting..." -ForegroundColor Yellow
black .
black --check .
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Black formatting failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[PASS] Black formatting passed" -ForegroundColor Green

# 4. Final linting check with flake8
Write-Host "`n[4/6] Final linting check with flake8..." -ForegroundColor Yellow
flake8 src/ --ignore=E501
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] flake8 linting failed after auto-fixes!" -ForegroundColor Red
    Write-Host "[INFO] Some issues may need manual fixing" -ForegroundColor Yellow
    exit 1
}
Write-Host "[PASS] flake8 linting passed" -ForegroundColor Green

# 5. Run tests
Write-Host "`n[5/6] Running tests..." -ForegroundColor Yellow
pytest --maxfail=1 -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Tests failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[PASS] Tests passed" -ForegroundColor Green

# 6. Check if docs build
Write-Host "`n[6/6] Building documentation..." -ForegroundColor Yellow
$env:JUPYTER_PLATFORM_DIRS = "1"
mkdocs build --site-dir site
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Documentation build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "[PASS] Documentation build passed" -ForegroundColor Green

Write-Host "`n[SUCCESS] All pre-commit checks passed! Safe to commit." -ForegroundColor Green 