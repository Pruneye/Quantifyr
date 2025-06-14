# Quantifyr

Quantifyr is a hybrid classical–quantum molecular autoencoder and property‑prediction platform.
It provides:

- **Classical AE** (PyTorch) baseline
- **Quantum AE** (PennyLane/Qiskit) and **QVAE**
- **Hybrid GNN→Quantum** pipelines
- Extensive **2D/3D/interactive visualizations** (Matplotlib/Plotly/Dash)
- Automated **CI (lint, tests)** and **Docs** via MkDocs + GitHub Pages

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
