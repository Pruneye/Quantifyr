"""
Molecular data processing utilities for Quantifyr.

This module provides functions for parsing molecules, extracting features,
and converting to graph representations for classical and quantum models.

This is the main entry point that imports from specialized modules:
- parsing: SMILES parsing and molecule creation
- features: Atomic and bond feature extraction
- graphs: Graph construction for GNNs
- properties: Molecular descriptor calculation
- datasets: Batch processing and dataset creation
"""

# Import all functionality from specialized modules
from .parsing import (
    MoleculeConfig,
    parse_smiles,
    parse_sdf_file,
)

from .features import (
    get_element_symbol,
    extract_atom_features,
    extract_bond_features,
)

from .graphs import (
    mol_to_graph,
)

from .properties import (
    compute_molecular_descriptors,
)

from .datasets import (
    load_molecule_dataset,
    create_molecular_dataframe,
)

# Re-export all functions for backward compatibility
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
