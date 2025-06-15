"""
Molecular data processing utilities for Quantifyr.

This package provides comprehensive molecular data processing capabilities:
- parsing: SMILES parsing and molecule creation
- features: Atomic and bond feature extraction
- graphs: Graph construction for GNNs
- properties: Molecular descriptor calculation
- datasets: Batch processing and dataset creation
"""

# Import all functionality from the main module (which imports from submodules)
from .data_utils import (
    # Configuration
    MoleculeConfig,
    # Parsing
    parse_smiles,
    parse_sdf_file,
    # Features
    get_element_symbol,
    extract_atom_features,
    extract_bond_features,
    # Graphs
    mol_to_graph,
    # Properties
    compute_molecular_descriptors,
    # Datasets
    load_molecule_dataset,
    create_molecular_dataframe,
)

__all__ = [
    # Configuration
    "MoleculeConfig",
    # Parsing
    "parse_smiles",
    "parse_sdf_file",
    # Features
    "get_element_symbol",
    "extract_atom_features",
    "extract_bond_features",
    # Graphs
    "mol_to_graph",
    # Properties
    "compute_molecular_descriptors",
    # Datasets
    "load_molecule_dataset",
    "create_molecular_dataframe",
]
