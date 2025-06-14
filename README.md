# Quantifyr

[![CI](https://github.com/Pruneye/Quantifyr/workflows/CI/badge.svg)](https://github.com/Pruneye/Quantifyr/actions)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://Pruneye.github.io/Quantifyr/)

**A hybrid classical-quantum molecular machine learning platform**. Transform molecules into quantum-ready representations with interactive visualizations and robust ML pipelines.

## Installation

```bash
git clone https://github.com/Pruneye/Quantifyr.git
cd Quantifyr
pip install -r requirements.txt
```

## Quick Start

```python
from data_utils import parse_smiles, mol_to_graph, create_molecular_dataframe
from viz import draw_molecule_2d, create_3d_conformer_plot

# Parse and analyze molecules
mol = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # caffeine
graph = mol_to_graph(mol)  # PyTorch Geometric graph
img = draw_molecule_2d(mol)  # 2D structure

# Batch analysis with properties
smiles_list = ["CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"]
df = create_molecular_dataframe(smiles_list)
# Returns DataFrame with MW, LogP, TPSA, etc.

# Interactive 3D visualization
fig = create_3d_conformer_plot(mol)
fig.show()  # Rotatable 3D molecular model
```

## Examples

- **Demo Script**: `python examples/molecular_analysis_demo.py`
- **Interactive Tutorial**: `notebooks/exploratory/01_molecular_data_processing.ipynb`

## Documentation

**[Full Documentation](https://Pruneye.github.io/Quantifyr/)**

## Development

```bash
# Run pre-commit checks
.\scripts\pre-commit-check.ps1  # Windows
./scripts/pre-commit-check.sh   # Linux/Mac

# Run tests
pytest --disable-warnings -q
```

## Roadmap

- **Stage 1**: Molecular data processing and visualization (Complete)
- **Stage 2**: Classical autoencoders (Planned)
- **Stage 3**: Quantum autoencoders (Planned)
- **Stage 4**: Hybrid GNN-Quantum pipelines (Planned)
