"""
Molecular property calculation utilities.

This module provides functions for computing molecular descriptors
and properties for QSAR analysis and property prediction.
"""

from typing import Dict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def compute_molecular_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute drug-like molecular descriptors for QSAR/property prediction.

    Calculated properties:
    - molecular_weight - Molecular weight in Daltons
    - logp - Lipophilicity (partition coefficient)
    - tpsa - Topological polar surface area (Ų)
    - num_rotatable_bonds - Flexibility measure
    - num_hbd - Hydrogen bond donors (NH, OH groups)
    - num_hba - Hydrogen bond acceptors (N, O atoms)
    - num_rings - Ring count (important for rigidity)
    - num_aromatic_rings - Aromatic ring count

    Args:
        mol: RDKit molecule object (from parse_smiles)

    Returns:
        Dictionary mapping property names to float values

    Examples:
        >>> mol = parse_smiles("CCO")  # ethanol
        >>> props = compute_molecular_descriptors(mol)
        >>> print(f"MW: {props['molecular_weight']:.1f} Da")  # MW: 46.1 Da
        >>> print(f"LogP: {props['logp']:.2f}")  # LogP: -0.31 (hydrophilic)
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
