"""
SMILES parsing and molecule creation utilities.

This module handles parsing SMILES strings into RDKit molecule objects
with comprehensive error handling and configuration options.
"""

from typing import List, Optional, Union
from pathlib import Path
from rdkit import Chem
from dataclasses import dataclass


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
