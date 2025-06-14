import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem
from PIL import Image

from viz import (
    draw_molecule_2d,
    plot_molecular_properties,
    plot_property_distribution,
    plot_molecular_network,
    create_3d_conformer_plot,
    plot_feature_importance,
    create_molecular_dashboard,
)
from data_utils import (
    parse_smiles,
    create_molecular_dataframe,
    load_molecule_dataset,
)


class TestDrawMolecule2D:
    """Test 2D molecular drawing functionality."""

    def test_draw_valid_molecule(self):
        """Test drawing a valid molecule."""
        mol = parse_smiles("CCO")
        img = draw_molecule_2d(mol)

        assert isinstance(img, Image.Image)
        assert img.size == (300, 300)  # Default size

    def test_draw_custom_size(self):
        """Test drawing with custom size."""
        mol = parse_smiles("CCO")
        img = draw_molecule_2d(mol, size=(500, 400))

        assert img.size == (500, 400)

    def test_draw_none_molecule(self):
        """Test drawing None molecule raises error."""
        with pytest.raises(ValueError, match="Cannot draw None molecule"):
            draw_molecule_2d(None)


class TestPlotMolecularProperties:
    """Test molecular property plotting."""

    def test_plot_properties_valid_data(self):
        """Test plotting with valid molecular data."""
        smiles_list = ["CCO", "CC", "CCC", "CCCC"]
        df = create_molecular_dataframe(smiles_list)

        properties = ["molecular_weight", "logp"]
        fig = plot_molecular_properties(df, properties)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_properties_insufficient_properties(self):
        """Test plotting with insufficient properties."""
        df = create_molecular_dataframe(["CCO"])

        with pytest.raises(ValueError, match="Need at least 2 properties"):
            plot_molecular_properties(df, ["molecular_weight"])

    def test_plot_properties_no_valid_molecules(self):
        """Test plotting with no valid molecules."""
        # Create DataFrame with only invalid molecules
        df = pd.DataFrame(
            {
                "smiles": ["invalid1", "invalid2"],
                "valid": [False, False],
                "molecular_weight": [np.nan, np.nan],
                "logp": [np.nan, np.nan],
            }
        )

        with pytest.raises(ValueError, match="No valid molecules to plot"):
            plot_molecular_properties(df, ["molecular_weight", "logp"])


class TestPlotPropertyDistribution:
    """Test property distribution plotting."""

    def test_plot_distribution_valid_property(self):
        """Test plotting distribution of valid property."""
        smiles_list = ["CCO", "CC", "CCC", "CCCC"]
        df = create_molecular_dataframe(smiles_list)

        fig = plot_property_distribution(df, "molecular_weight")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Histogram + box plot

    def test_plot_distribution_invalid_property(self):
        """Test plotting distribution of non-existent property."""
        df = create_molecular_dataframe(["CCO"])

        with pytest.raises(ValueError, match="Property 'nonexistent' not found"):
            plot_property_distribution(df, "nonexistent")

    def test_plot_distribution_no_valid_data(self):
        """Test plotting with no valid data."""
        df = pd.DataFrame(
            {"smiles": ["invalid"], "valid": [False], "molecular_weight": [np.nan]}
        )

        with pytest.raises(ValueError, match="Property 'molecular_weight' not found"):
            plot_property_distribution(df, "molecular_weight")


class TestPlotMolecularNetwork:
    """Test molecular network visualization."""

    def test_plot_network_valid_graphs(self):
        """Test plotting molecular networks."""
        smiles_list = ["CCO", "CC"]
        graphs, _ = load_molecule_dataset(smiles_list)

        fig = plot_molecular_network(graphs)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_network_empty_graphs(self):
        """Test plotting with empty graph list."""
        with pytest.raises(ValueError, match="No graphs provided"):
            plot_molecular_network([])

    def test_plot_network_max_molecules(self):
        """Test plotting with max molecules limit."""
        smiles_list = ["CCO", "CC", "CCC", "CCCC"] * 5  # 20 molecules
        graphs, _ = load_molecule_dataset(smiles_list)

        fig = plot_molecular_network(graphs, max_molecules=3)

        assert isinstance(fig, go.Figure)
        # Should limit to 3 molecules, so fewer traces than all molecules


class TestCreate3DConformerPlot:
    """Test 3D conformer visualization."""

    def test_plot_3d_valid_molecule(self):
        """Test 3D plotting of valid molecule."""
        mol = parse_smiles("CCO")
        fig = create_3d_conformer_plot(mol)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        # Should have 3D scatter plot
        assert any(isinstance(trace, go.Scatter3d) for trace in fig.data)

    def test_plot_3d_none_molecule(self):
        """Test 3D plotting of None molecule."""
        with pytest.raises(ValueError, match="Cannot plot None molecule"):
            create_3d_conformer_plot(None)


class TestPlotFeatureImportance:
    """Test feature importance plotting."""

    def test_plot_feature_importance_valid_data(self):
        """Test plotting feature importance."""
        features = np.array([0.5, 0.3, 0.8, 0.1])
        feature_names = ["Feature1", "Feature2", "Feature3", "Feature4"]

        fig = plot_feature_importance(features, feature_names)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)

    def test_plot_feature_importance_mismatched_lengths(self):
        """Test plotting with mismatched feature and name lengths."""
        features = np.array([0.5, 0.3])
        feature_names = ["Feature1", "Feature2", "Feature3"]

        with pytest.raises(
            ValueError, match="Features and feature names must have same length"
        ):
            plot_feature_importance(features, feature_names)


class TestCreateMolecularDashboard:
    """Test molecular dashboard creation."""

    def test_create_dashboard_valid_data(self):
        """Test creating dashboard with valid data."""
        smiles_list = ["CCO", "CC", "CCC", "CCCC"]
        df = create_molecular_dataframe(smiles_list)
        molecules = [parse_smiles(s) for s in smiles_list]

        dashboard = create_molecular_dashboard(df, molecules)

        assert isinstance(dashboard, dict)
        assert len(dashboard) > 0

        # Should contain various plot types
        for key, fig in dashboard.items():
            assert isinstance(fig, go.Figure)

    def test_create_dashboard_empty_molecules(self):
        """Test creating dashboard with empty molecule list."""
        df = create_molecular_dataframe(["CCO", "CC"])
        molecules = []

        dashboard = create_molecular_dashboard(df, molecules)

        assert isinstance(dashboard, dict)
        # Should still create some plots even without 3D structure

    def test_create_dashboard_minimal_data(self):
        """Test creating dashboard with minimal data."""
        smiles_list = ["CCO"]
        df = create_molecular_dataframe(smiles_list)
        molecules = [parse_smiles(smiles_list[0])]

        dashboard = create_molecular_dashboard(df, molecules)

        assert isinstance(dashboard, dict)


# Integration tests combining data processing and visualization
class TestDataVizIntegration:
    """Test integration between data processing and visualization."""

    def test_end_to_end_molecular_analysis(self):
        """Test complete molecular analysis pipeline."""
        # Sample molecules with diverse properties
        smiles_list = [
            "CCO",  # ethanol
            "CC(C)C",  # isobutane
            "c1ccccc1",  # benzene
            "CCN(CC)CC",  # triethylamine
            "C(=O)O",  # formic acid
        ]

        # Create molecular dataframe
        df = create_molecular_dataframe(smiles_list)

        # Load as graph dataset
        graphs, _ = load_molecule_dataset(smiles_list)

        # Create visualizations
        molecules = [parse_smiles(s) for s in smiles_list]

        # Test that all visualization functions work with real data
        assert len(df) == 5
        assert df["valid"].sum() == 5  # All should be valid

        # Test 2D drawing
        for mol in molecules:
            if mol is not None:
                img = draw_molecule_2d(mol)
                assert isinstance(img, Image.Image)

        # Test property plotting
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != "valid"]

        if len(numeric_cols) >= 2:
            fig = plot_molecular_properties(df, numeric_cols[:2])
            assert isinstance(fig, go.Figure)

        # Test network plotting
        fig = plot_molecular_network(graphs, max_molecules=3)
        assert isinstance(fig, go.Figure)

        # Test 3D plotting
        if molecules and molecules[0] is not None:
            fig = create_3d_conformer_plot(molecules[0])
            assert isinstance(fig, go.Figure)

    def test_property_analysis_workflow(self):
        """Test property analysis workflow."""
        smiles_list = ["CCO", "CC", "CCC", "CCCC", "CCCCC"]
        df = create_molecular_dataframe(smiles_list)

        # Test property distribution analysis
        fig = plot_property_distribution(df, "molecular_weight")
        assert isinstance(fig, go.Figure)

        # Test scatter matrix
        properties = ["molecular_weight", "logp", "tpsa"]
        fig = plot_molecular_properties(df, properties)
        assert isinstance(fig, go.Figure)

        # Verify data quality
        assert df["valid"].all()  # All molecules should be valid
        assert df["molecular_weight"].min() > 0  # Positive molecular weights
