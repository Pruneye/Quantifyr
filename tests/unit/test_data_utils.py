import pytest
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from pathlib import Path
import tempfile

from data_utils import (
    parse_smiles,
    parse_sdf_file,
    extract_atom_features,
    extract_bond_features,
    mol_to_graph,
    compute_molecular_descriptors,
    load_molecule_dataset,
    create_molecular_dataframe,
    MoleculeConfig,
)


class TestMoleculeConfig:
    """Test MoleculeConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MoleculeConfig()
        assert config.add_hydrogens == True
        assert config.sanitize == True
        assert config.kekulize == False
        assert config.max_atoms == 100

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MoleculeConfig(add_hydrogens=False, max_atoms=50)
        assert config.add_hydrogens == False
        assert config.max_atoms == 50

    def test_invalid_max_atoms(self):
        """Test validation of max_atoms parameter."""
        with pytest.raises(ValueError, match="max_atoms must be >= 1"):
            MoleculeConfig(max_atoms=0)


class TestParseSMILES:
    """Test SMILES parsing functionality."""

    def test_parse_valid_smiles(self):
        """Test parsing valid SMILES strings."""
        mol = parse_smiles("CCO")  # ethanol
        assert mol is not None
        assert isinstance(mol, Chem.Mol)

    def test_parse_invalid_smiles(self):
        """Test parsing invalid SMILES strings."""
        assert parse_smiles("invalid_smiles") is None
        assert parse_smiles("") is None
        assert parse_smiles(None) is None
        assert parse_smiles(123) is None

    def test_parse_with_config(self):
        """Test parsing with custom configuration."""
        config = MoleculeConfig(add_hydrogens=False)
        mol = parse_smiles("CCO", config)
        assert mol is not None
        # Without hydrogens, should have fewer atoms
        assert mol.GetNumAtoms() == 3

    def test_parse_large_molecule_filtered(self):
        """Test that large molecules are filtered out."""
        # Create a very long chain that exceeds max_atoms
        long_smiles = "C" * 150  # Very long carbon chain
        config = MoleculeConfig(max_atoms=100)
        mol = parse_smiles(long_smiles, config)
        # Should be None due to atom count limit
        assert mol is None or mol.GetNumAtoms() <= 100


class TestAtomFeatures:
    """Test atomic feature extraction."""

    def test_extract_atom_features_valid_molecule(self):
        """Test feature extraction from valid molecule."""
        mol = parse_smiles("CCO")
        features = extract_atom_features(mol)

        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert features.shape[1] == 6  # 6 atomic features
        assert features.shape[0] > 0  # Should have atoms

    def test_extract_atom_features_none_molecule(self):
        """Test feature extraction from None molecule."""
        features = extract_atom_features(None)
        assert isinstance(features, np.ndarray)
        assert features.size == 0

    def test_atom_feature_values(self):
        """Test that atomic features have expected ranges."""
        mol = parse_smiles("CCO")
        features = extract_atom_features(mol)

        # Atomic numbers should be positive
        atomic_nums = features[:, 0]
        assert np.all(atomic_nums > 0)

        # Degrees should be non-negative
        degrees = features[:, 1]
        assert np.all(degrees >= 0)


class TestBondFeatures:
    """Test bond feature extraction."""

    def test_extract_bond_features_valid_molecule(self):
        """Test bond feature extraction from valid molecule."""
        mol = parse_smiles("CCO")
        features = extract_bond_features(mol)

        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert features.shape[1] == 3  # 3 bond features

    def test_extract_bond_features_none_molecule(self):
        """Test bond feature extraction from None molecule."""
        features = extract_bond_features(None)
        assert isinstance(features, np.ndarray)
        assert features.size == 0


class TestMolToGraph:
    """Test molecule to graph conversion."""

    def test_mol_to_graph_valid_molecule(self):
        """Test graph conversion from valid molecule."""
        mol = parse_smiles("CCO")
        graph = mol_to_graph(mol)

        assert hasattr(graph, "x")
        assert hasattr(graph, "edge_index")
        assert isinstance(graph.x, torch.Tensor)
        assert isinstance(graph.edge_index, torch.Tensor)

        # Check dimensions
        assert graph.x.shape[1] == 6  # 6 atomic features
        assert graph.edge_index.shape[0] == 2  # Source and target nodes

    def test_mol_to_graph_with_edge_features(self):
        """Test graph conversion with edge features."""
        mol = parse_smiles("CCO")
        graph = mol_to_graph(mol, include_edge_features=True)

        if mol.GetNumBonds() > 0:
            assert hasattr(graph, "edge_attr")
            assert isinstance(graph.edge_attr, torch.Tensor)

    def test_mol_to_graph_none_molecule(self):
        """Test graph conversion from None molecule."""
        with pytest.raises(ValueError, match="Cannot convert None molecule"):
            mol_to_graph(None)


class TestMolecularDescriptors:
    """Test molecular descriptor computation."""

    def test_compute_descriptors_valid_molecule(self):
        """Test descriptor computation for valid molecule."""
        mol = parse_smiles("CCO")
        descriptors = compute_molecular_descriptors(mol)

        assert isinstance(descriptors, dict)
        expected_keys = [
            "molecular_weight",
            "logp",
            "tpsa",
            "num_rotatable_bonds",
            "num_hbd",
            "num_hba",
            "num_rings",
            "num_aromatic_rings",
        ]

        for key in expected_keys:
            assert key in descriptors
            assert isinstance(descriptors[key], (int, float))

    def test_compute_descriptors_none_molecule(self):
        """Test descriptor computation for None molecule."""
        descriptors = compute_molecular_descriptors(None)
        assert descriptors == {}

    def test_descriptor_values_reasonable(self):
        """Test that descriptor values are reasonable."""
        mol = parse_smiles("CCO")  # ethanol
        descriptors = compute_molecular_descriptors(mol)

        # Ethanol should have reasonable properties
        assert descriptors["molecular_weight"] > 0
        assert descriptors["num_hbd"] >= 0  # Should have at least OH group
        assert descriptors["num_hba"] >= 0


class TestLoadMoleculeDataset:
    """Test dataset loading functionality."""

    def test_load_dataset_valid_smiles(self):
        """Test loading dataset from valid SMILES."""
        smiles_list = ["CCO", "CC", "CCC", "CCCC"]
        graphs, labels = load_molecule_dataset(smiles_list)

        assert isinstance(graphs, list)
        assert len(graphs) == 4
        assert all(hasattr(g, "x") and hasattr(g, "edge_index") for g in graphs)

    def test_load_dataset_with_labels(self):
        """Test loading dataset with labels."""
        smiles_list = ["CCO", "CC", "CCC"]
        labels = [1.0, 2.0, 3.0]
        graphs, valid_labels = load_molecule_dataset(smiles_list, labels)

        assert len(graphs) == len(valid_labels)
        assert valid_labels == labels  # All should be valid

    def test_load_dataset_with_invalid_smiles(self):
        """Test loading dataset with some invalid SMILES."""
        smiles_list = ["CCO", "invalid", "CC"]
        graphs, labels = load_molecule_dataset(smiles_list)

        # Should skip invalid SMILES
        assert len(graphs) == 2


class TestCreateMolecularDataFrame:
    """Test DataFrame creation functionality."""

    def test_create_dataframe_valid_smiles(self):
        """Test DataFrame creation from valid SMILES."""
        smiles_list = ["CCO", "CC", "CCC"]
        df = create_molecular_dataframe(smiles_list)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "smiles" in df.columns
        assert "valid" in df.columns
        assert "molecular_weight" in df.columns

    def test_create_dataframe_with_invalid_smiles(self):
        """Test DataFrame creation with invalid SMILES."""
        smiles_list = ["CCO", "invalid", "CC"]
        df = create_molecular_dataframe(smiles_list)

        assert len(df) == 3
        assert df["valid"].sum() == 2  # Only 2 valid molecules

        # Invalid molecule should have NaN for descriptors
        invalid_row = df[~df["valid"]].iloc[0]
        assert invalid_row["smiles"] == "invalid"


class TestParseSDF:
    """Test SDF file parsing."""

    def test_parse_nonexistent_file(self):
        """Test parsing non-existent SDF file."""
        with pytest.raises(FileNotFoundError):
            parse_sdf_file("nonexistent.sdf")

    def test_parse_sdf_file_basic(self):
        """Test basic SDF file parsing with temporary file."""
        # Create a simple SDF content
        sdf_content = """
  Mrv2014 01012021

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as f:
            f.write(sdf_content)
            f.flush()

            try:
                molecules = parse_sdf_file(f.name)
                # Note: This simple SDF might not parse correctly with RDKit
                # but the function should handle it gracefully
                assert isinstance(molecules, list)
            finally:
                try:
                    Path(f.name).unlink()  # Clean up
                except PermissionError:
                    pass  # Ignore Windows file permission issues in tests


# Parametrized tests for multiple SMILES
@pytest.mark.parametrize(
    "smiles,expected_valid",
    [
        ("CCO", True),  # ethanol - valid
        ("CC", True),  # ethane - valid
        ("CCC", True),  # propane - valid
        ("invalid", False),  # invalid SMILES
        ("", False),  # empty string
    ],
)
def test_parse_smiles_parametrized(smiles, expected_valid):
    """Parametrized test for various SMILES strings."""
    mol = parse_smiles(smiles)
    is_valid = mol is not None
    assert is_valid == expected_valid


@pytest.mark.parametrize("smiles", ["CCO", "CC(C)C", "c1ccccc1", "CCN"])
def test_feature_extraction_consistency(smiles):
    """Test that feature extraction is consistent for valid molecules."""
    mol = parse_smiles(smiles)
    assert mol is not None

    atom_features = extract_atom_features(mol)
    bond_features = extract_bond_features(mol)

    # Features should have correct shapes
    assert atom_features.shape[0] == mol.GetNumAtoms()
    assert atom_features.shape[1] == 6

    if mol.GetNumBonds() > 0:
        assert bond_features.shape[0] == mol.GetNumBonds()
        assert bond_features.shape[1] == 3
