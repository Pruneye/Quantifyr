"""
Molecular feature extraction utilities.

This module provides functions for extracting atomic and bond features
from molecules for machine learning models.
"""

import numpy as np
from rdkit import Chem


def get_element_symbol(atomic_num: int) -> str:
    """Get element symbol from atomic number using RDKit's periodic table."""
    try:
        return Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
    except BaseException:
        return f"#{atomic_num}"


def extract_atom_features(mol: Chem.Mol) -> np.ndarray:
    """
    Extract comprehensive atomic features from molecule for ML models.

    **Feature vector (6 features per atom):**
    1. **Atomic number** - Element identity (C=6, N=7, O=8, etc.)
    2. **Degree** - Number of bonded neighbors
    3. **Formal charge** - Charge on atom
    4. **Hybridization** - SP/SP2/SP3 hybridization state
    5. **Aromaticity** - Is atom in aromatic ring (1/0)
    6. **Hydrogen count** - Number of attached hydrogens

    Args:
        mol: RDKit molecule object (from parse_smiles)

    Returns:
        Array of shape (n_atoms, 6) with atomic features

    Examples:
        >>> mol = parse_smiles("CCO")  # ethanol
        >>> features = extract_atom_features(mol)
        >>> print(f"Shape: {features.shape}")  # (9, 6) - 9 atoms, 6 features
        >>> # First atom (Carbon): [6.0, 4.0, 0.0, 3.0, 0.0, 3.0]
        >>> # Element C, degree 4, no charge, SP3, not aromatic, 3 H atoms
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
    Extract bond features from molecule for graph neural networks.

    **Feature vector (3 features per bond):**
    1. **Bond type** - Single=1, Double=2, Triple=3, Aromatic=1.5
    2. **Conjugation** - Is bond part of conjugated system (1/0)
    3. **Ring membership** - Is bond in a ring (1/0)

    Args:
        mol: RDKit molecule object (from parse_smiles)

    Returns:
        Array of shape (n_bonds, 3) with bond features

    Examples:
        >>> mol = parse_smiles("c1ccccc1")  # benzene
        >>> features = extract_bond_features(mol)
        >>> print(f"Benzene has {len(features)} bonds")  # 6 bonds in ring
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
