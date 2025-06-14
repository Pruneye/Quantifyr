# Quantifyr

**A hybrid classical-quantum molecular machine learning platform**.
Transform molecules into quantum-ready representations with interactive visualizations and robust ML pipelines.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from data_utils import parse_smiles, mol_to_graph, create_molecular_dataframe
from viz import draw_molecule_2d, create_3d_conformer_plot

# Parse molecule from SMILES
mol = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # caffeine

# Create visualizations
img = draw_molecule_2d(mol)
fig = create_3d_conformer_plot(mol)

# Convert to ML-ready graph
graph = mol_to_graph(mol)  # PyTorch Geometric format

# Batch analysis with properties
df = create_molecular_dataframe(["CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"])
```

## Core Modules

### data_utils

- `parse_smiles()` - SMILES string parsing and validation
- `mol_to_graph()` - Convert to PyTorch Geometric graphs
- `create_molecular_dataframe()` - Batch property analysis
- `compute_molecular_descriptors()` - Calculate MW, LogP, TPSA, etc.

### viz

- `draw_molecule_2d()` - Generate 2D molecular structures
- `create_3d_conformer_plot()` - Interactive 3D molecular models
- `plot_molecular_properties()` - Property correlation analysis
- `create_molecular_dashboard()` - Comprehensive analysis dashboards

## Tutorials

- [Data Processing](tutorials/data_processing.md) - Complete molecular data processing guide
- [Visualization](tutorials/visualization.md) - Interactive molecular visualizations
- [API Reference](tutorials/api-reference.md) - Detailed function documentation

## Development Status

- **Stage 1**: Molecular data processing and visualization (Complete)
- **Stage 2**: Classical autoencoders (Planned)
- **Stage 3**: Quantum autoencoders (Planned)
- **Stage 4**: Hybrid GNN-Quantum pipelines (Planned)
