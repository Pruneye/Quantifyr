"""
Molecular data processing utilities for Quantifyr.

This module provides functions for parsing molecules, extracting features,
and converting to graph representations for classical and quantum models.
"""

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdmolops
from dataclasses import dataclass
import pandas as pd
from pathlib import Path


@dataclass
class MoleculeConfig:
    """Configuration for molecular data processing."""

    add_hydrogens: bool = True
    sanitize: bool = True
    kekulize: bool = False
    max_atoms: int = 100

    def __post_init__(self):
        if self.max_atoms < 1:
            raise ValueError("max_atoms must be >= 1")


def parse_smiles(
    smiles: str, config: Optional[MoleculeConfig] = None
) -> Optional[Chem.Mol]:
    """
    Parse a SMILES string into an RDKit molecule object.

    Args:
        smiles: SMILES string representation of molecule
        config: Configuration for molecule processing

    Returns:
        RDKit molecule object or None if parsing fails

    Examples:
        >>> mol = parse_smiles("CCO")  # ethanol
        >>> mol.GetNumAtoms()
        3

        >>> parse_smiles("invalid") is None
        True
    """
    if not smiles or not isinstance(smiles, str):
        return None

    config = config or MoleculeConfig()

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if config.sanitize:
            Chem.SanitizeMol(mol)

        if config.add_hydrogens:
            mol = Chem.AddHs(mol)

        if config.kekulize:
            Chem.Kekulize(mol)

        # Check atom count limit
        if mol.GetNumAtoms() > config.max_atoms:
            return None

        return mol

    except Exception:
        return None


def parse_sdf_file(
    filepath: Union[str, Path], config: Optional[MoleculeConfig] = None
) -> List[Chem.Mol]:
    """
    Parse molecules from an SDF file.

    Args:
        filepath: Path to SDF file
        config: Configuration for molecule processing

    Returns:
        List of successfully parsed RDKit molecule objects

    Examples:
        >>> molecules = parse_sdf_file("molecules.sdf")
        >>> len(molecules)
        42
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SDF file not found: {filepath}")

    config = config or MoleculeConfig()
    molecules = []

    try:
        supplier = Chem.SDMolSupplier(str(filepath))
        for mol in supplier:
            if mol is not None:
                if config.sanitize:
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        continue

                if config.add_hydrogens:
                    mol = Chem.AddHs(mol)

                if mol.GetNumAtoms() <= config.max_atoms:
                    molecules.append(mol)

    except Exception as e:
        raise RuntimeError(f"Error reading SDF file: {e}")

    return molecules


def extract_atom_features(mol: Chem.Mol) -> np.ndarray:
    """
    Extract atomic features from molecule.

    Features include: atomic number, degree, formal charge, hybridization,
    aromaticity, and number of hydrogens.

    Args:
        mol: RDKit molecule object

    Returns:
        Array of shape (n_atoms, n_features) with atomic features

    Examples:
        >>> mol = parse_smiles("CCO")
        >>> features = extract_atom_features(mol)
        >>> features.shape
        (9, 6)  # 9 atoms (with H), 6 features
    """
    if mol is None:
        return np.array([])

    features = []
    for atom in mol.GetAtoms():
        atom_features = [
            atom.GetAtomicNum(),  # Atomic number
            atom.GetDegree(),  # Number of neighbors
            atom.GetFormalCharge(),  # Formal charge
            int(atom.GetHybridization()),  # Hybridization
            int(atom.GetIsAromatic()),  # Aromaticity
            atom.GetTotalNumHs(),  # Number of hydrogens
        ]
        features.append(atom_features)

    return np.array(features, dtype=np.float32)


def extract_bond_features(mol: Chem.Mol) -> np.ndarray:
    """
    Extract bond features from molecule.

    Features include: bond type, conjugation, and ring membership.

    Args:
        mol: RDKit molecule object

    Returns:
        Array of shape (n_bonds, n_features) with bond features
    """
    if mol is None:
        return np.array([])

    features = []
    for bond in mol.GetBonds():
        bond_features = [
            int(bond.GetBondType()),  # Bond type (single, double, etc.)
            int(bond.GetIsConjugated()),  # Conjugation
            int(bond.IsInRing()),  # Ring membership
        ]
        features.append(bond_features)

    return np.array(features, dtype=np.float32)


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


def compute_molecular_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute molecular descriptors for property prediction.

    Args:
        mol: RDKit molecule object

    Returns:
        Dictionary of molecular descriptors

    Examples:
        >>> mol = parse_smiles("CCO")
        >>> descriptors = compute_molecular_descriptors(mol)
        >>> "molecular_weight" in descriptors
        True
    """
    if mol is None:
        return {}

    descriptors = {
        "molecular_weight": rdMolDescriptors.CalcExactMolWt(mol),
        "logp": rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "num_hbd": rdMolDescriptors.CalcNumHBD(mol),  # Hydrogen bond donors
        "num_hba": rdMolDescriptors.CalcNumHBA(mol),  # Hydrogen bond acceptors
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }

    return descriptors


def load_molecule_dataset(
    smiles_list: List[str],
    labels: Optional[List[float]] = None,
    config: Optional[MoleculeConfig] = None,
) -> Tuple[List[Data], List[float]]:
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
        mol = parse_smiles(smiles, config)
        if mol is not None:
            try:
                graph = mol_to_graph(mol)
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
