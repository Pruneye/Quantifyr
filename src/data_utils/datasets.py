"""
Dataset creation and batch processing utilities.

This module provides functions for loading and processing multiple molecules
into datasets suitable for machine learning workflows.
"""

from typing import List, Optional, Tuple
import pandas as pd
from .parsing import parse_smiles, MoleculeConfig
from .graphs import mol_to_graph
from .properties import compute_molecular_descriptors


def load_molecule_dataset(
    smiles_list: List[str],
    labels: Optional[List[float]] = None,
    config: Optional[MoleculeConfig] = None,
) -> Tuple[List, List[float]]:
    """
    Load a dataset of molecules from SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        labels: Optional list of labels for supervised learning
        config: Configuration for molecule processing

    Returns:
        Tuple of (graph_data_list, valid_labels)

    Examples:
        >>> smiles = ["CCO", "CC", "CCC"]
        >>> graphs, _ = load_molecule_dataset(smiles)
        >>> len(graphs)
        3
    """
    config = config or MoleculeConfig()
    graphs = []
    valid_labels = []

    for i, smiles in enumerate(smiles_list):
        try:
            mol = parse_smiles(smiles, config)
            if mol is not None:
                graph = mol_to_graph(mol)
                if graph is not None and hasattr(graph, "x") and graph.x.shape[0] > 0:
                    graphs.append(graph)
                    if labels is not None:
                        valid_labels.append(labels[i])
        except Exception:
            continue  # Skip problematic molecules

    return graphs, valid_labels if labels is not None else []


def create_molecular_dataframe(smiles_list: List[str]) -> pd.DataFrame:
    """
    Create a pandas DataFrame with molecular properties from SMILES.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        DataFrame with SMILES and computed molecular descriptors
    """
    data = []

    for smiles in smiles_list:
        mol = parse_smiles(smiles)
        row = {"smiles": smiles}

        if mol is not None:
            descriptors = compute_molecular_descriptors(mol)
            row.update(descriptors)
            row["valid"] = True
        else:
            row["valid"] = False

        data.append(row)

    return pd.DataFrame(data)
