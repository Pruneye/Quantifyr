# Quantifyr

[![CI](https://github.com/Pruneye/Quantifyr/workflows/CI/badge.svg)](https://github.com/Pruneye/Quantifyr/actions)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://Pruneye.github.io/Quantifyr/)

Quantifyr is a hybrid classical–quantum molecular autoencoder and property‑prediction platform.
It provides:

- **Classical AE** (PyTorch) baseline
- **Quantum AE** (PennyLane/Qiskit) and **QVAE**
- **Hybrid GNN→Quantum** pipelines
- Extensive **2D/3D/interactive visualizations** (Matplotlib/Plotly/Dash)
- Automated **CI (lint, tests)** and **Docs** via MkDocs + GitHub Pages

## 📚 Documentation

View the full documentation at: **[https://Pruneye.github.io/Quantifyr/](https://Pruneye.github.io/Quantifyr/)**

## Getting Started

```bash
# Setup virtual environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Development Workflow

### Before Committing

Always run the pre-commit checks to ensure CI passes:

```bash
# Windows
.\scripts\pre-commit-check.ps1

# Linux/Mac
./scripts/pre-commit-check.sh
```
