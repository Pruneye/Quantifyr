"""
Graph construction utilities for molecular data.

This module provides functions for converting molecules to graph representations
suitable for graph neural networks.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolops
from .features import extract_atom_features, extract_bond_features


def mol_to_graph(mol: Chem.Mol, include_edge_features: bool = True) -> Data:
    """
    Convert RDKit molecule to PyTorch Geometric graph.

    Args:
        mol: RDKit molecule object
        include_edge_features: Whether to include edge (bond) features

    Returns:
        PyTorch Geometric Data object with node and edge features

    Examples:
        >>> mol = parse_smiles("CCO")
        >>> graph = mol_to_graph(mol)
        >>> graph.x.shape
        torch.Size([9, 6])
        >>> graph.edge_index.shape
        torch.Size([2, 16])
    """
    if mol is None:
        raise ValueError("Cannot convert None molecule to graph")

    # Node features
    atom_features = extract_atom_features(mol)
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge indices (adjacency matrix)
    adjacency = rdmolops.GetAdjacencyMatrix(mol)
    edge_indices = np.array(np.nonzero(adjacency))
    edge_index = torch.tensor(edge_indices, dtype=torch.long)

    # Create graph data
    data = Data(x=x, edge_index=edge_index)

    # Add edge features if requested
    if include_edge_features and mol.GetNumBonds() > 0:
        bond_features = extract_bond_features(mol)
        # Match bond features to edge indices
        edge_attr = []
        for i, j in edge_index.t():
            bond = mol.GetBondBetweenAtoms(int(i), int(j))
            if bond is not None:
                bond_idx = bond.GetIdx()
                edge_attr.append(bond_features[bond_idx])
            else:
                edge_attr.append([0, 0, 0])  # Default for missing bonds
        # Convert to numpy array first to avoid PyTorch warning
        edge_attr_array = np.array(edge_attr, dtype=np.float32)
        data.edge_attr = torch.tensor(edge_attr_array, dtype=torch.float)

    return data
