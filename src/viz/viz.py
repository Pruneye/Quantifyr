"""
Interactive molecular visualization utilities for Quantifyr.

This module provides 2D and 3D visualization functions for molecules,
molecular properties, and graph representations.
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data
import io
from PIL import Image


def draw_molecule_2d(mol: Chem.Mol, size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """
    Draw 2D structure of molecule using RDKit.

    Args:
        mol: RDKit molecule object
        size: Image size (width, height)

    Returns:
        PIL Image of molecular structure

    Examples:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> img = draw_molecule_2d(mol)
        >>> img.size
        (300, 300)
    """
    if mol is None:
        raise ValueError("Cannot draw None molecule")

    # Generate 2D coordinates if not present
    rdDepictor.Compute2DCoords(mol)

    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])

    # Draw molecule
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Convert to PIL Image
    img_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_data))

    return img


def plot_molecular_properties(df: pd.DataFrame, properties: List[str]) -> go.Figure:
    """
    Create interactive scatter matrix plot of molecular properties.

    Args:
        df: DataFrame with molecular properties
        properties: List of property names to plot

    Returns:
        Plotly figure with scatter matrix

    Examples:
        >>> df = create_molecular_dataframe(["CCO", "CC", "CCC"])
        >>> fig = plot_molecular_properties(df, ["molecular_weight", "logp"])
    """
    if len(properties) < 2:
        raise ValueError("Need at least 2 properties for scatter matrix")

    # Filter valid molecules
    valid_df = df[df["valid"]].copy()

    if valid_df.empty:
        raise ValueError("No valid molecules to plot")

    # Create scatter matrix
    fig = px.scatter_matrix(
        valid_df,
        dimensions=properties,
        title="Molecular Properties Scatter Matrix",
        hover_data=["smiles"],
    )

    fig.update_layout(height=600, width=800, showlegend=False)

    return fig


def plot_molecular_network(graphs: List[Data], max_molecules: int = 10) -> go.Figure:
    """
    Visualize molecular graphs as network plots.

    Args:
        graphs: List of PyTorch Geometric Data objects
        max_molecules: Maximum number of molecules to display

    Returns:
        Plotly figure with molecular networks
    """
    if not graphs:
        raise ValueError("No graphs provided")

    # Limit number of molecules for readability
    graphs = graphs[:max_molecules]

    # Calculate grid dimensions
    n_graphs = len(graphs)
    cols = min(5, n_graphs)
    rows = min(2, (n_graphs + cols - 1) // cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Molecule {i + 1}" for i in range(n_graphs)],
        specs=[[{"type": "scatter"}] * cols for _ in range(rows)],
    )

    for idx, graph in enumerate(graphs):
        row = idx // cols + 1
        col = idx % cols + 1

        # Convert to NetworkX
        G = nx.Graph()

        # Add nodes with features
        for i in range(graph.x.shape[0]):
            atomic_num = int(graph.x[i, 0])  # First feature is atomic number
            G.add_node(i, atomic_num=atomic_num)

        # Add edges
        edge_list = graph.edge_index.t().numpy()
        G.add_edges_from(edge_list)

        # Layout
        pos = nx.spring_layout(G, seed=42)

        # Node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_colors = [G.nodes[node]["atomic_num"] for node in G.nodes()]

        # Edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Add edge trace
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=2, color="gray"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Add node trace
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(
                    size=15,
                    color=node_colors,
                    colorscale="Viridis",
                    line=dict(width=2, color="black"),
                ),
                showlegend=False,
                hovertemplate="Atomic Number: %{marker.color}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Molecular Graph Networks",
        height=400 * min(2, len(graphs)),
        width=1200,
        showlegend=False,
    )

    return fig


def plot_property_distribution(df: pd.DataFrame, property_name: str) -> go.Figure:
    """
    Plot distribution of molecular property.

    Args:
        df: DataFrame with molecular properties
        property_name: Name of property to plot

    Returns:
        Plotly figure with histogram and box plot
    """
    valid_df = df[df["valid"]].copy()

    if valid_df.empty or property_name not in valid_df.columns:
        raise ValueError(f"Property '{property_name}' not found in valid data")

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[f"{property_name} Distribution", f"{property_name} Box Plot"],
        row_heights=[0.7, 0.3],
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=valid_df[property_name],
            nbinsx=30,
            name="Distribution",
            marker_color="lightblue",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # Box plot
    fig.add_trace(
        go.Box(x=valid_df[property_name], name="Box Plot", marker_color="lightcoral"),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"Analysis of {property_name}", height=500, width=800, showlegend=False
    )

    return fig


def create_3d_conformer_plot(mol: Chem.Mol) -> go.Figure:
    """
    Create 3D visualization of molecular conformer.

    Args:
        mol: RDKit molecule object with 3D coordinates

    Returns:
        Plotly 3D scatter plot of molecular structure
    """
    if mol is None:
        raise ValueError("Cannot plot None molecule")

    # Check if molecule has 3D coordinates
    if mol.GetNumConformers() == 0:
        # Generate 3D conformer if not present
        from rdkit.Chem import rdDistGeom

        rdDistGeom.EmbedMolecule(mol)

    conf = mol.GetConformer()

    # Extract atomic positions and numbers
    positions = []
    atomic_nums = []
    colors = []

    # Color mapping for common elements
    element_colors = {
        1: "white",  # Hydrogen
        6: "gray",  # Carbon
        7: "blue",  # Nitrogen
        8: "red",  # Oxygen
        9: "green",  # Fluorine
        15: "orange",  # Phosphorus
        16: "yellow",  # Sulfur
        17: "green",  # Chlorine
    }

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        atomic_num = atom.GetAtomicNum()
        atomic_nums.append(atomic_num)
        colors.append(element_colors.get(atomic_num, "purple"))

    positions = np.array(positions)

    # Create 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                marker=dict(size=8, color=colors, line=dict(width=2, color="black")),
                text=[
                    f"Atom {i + 1}: {atomic_nums[i]}" for i in range(len(atomic_nums))
                ],
                hovertemplate="%{text}<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
            )
        ]
    )

    # Add bonds as lines
    bond_traces = []
    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()

        pos1 = positions[atom1_idx]
        pos2 = positions[atom2_idx]

        bond_traces.extend(
            [
                go.Scatter3d(
                    x=[pos1[0], pos2[0], None],
                    y=[pos1[1], pos2[1], None],
                    z=[pos1[2], pos2[2], None],
                    mode="lines",
                    line=dict(color="gray", width=4),
                    showlegend=False,
                    hoverinfo="skip",
                )
            ]
        )

    for trace in bond_traces:
        fig.add_trace(trace)

    fig.update_layout(
        title="3D Molecular Structure",
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode="cube",
        ),
        width=800,
        height=600,
        showlegend=False,
    )

    return fig


def plot_feature_importance(
    features: np.ndarray, feature_names: List[str]
) -> go.Figure:
    """
    Plot feature importance for molecular features.

    Args:
        features: Feature importance values
        feature_names: Names of features

    Returns:
        Plotly bar chart of feature importance
    """
    if len(features) != len(feature_names):
        raise ValueError("Features and feature names must have same length")

    # Sort by importance
    sorted_indices = np.argsort(features)[::-1]
    sorted_features = features[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]

    fig = go.Figure(
        data=[
            go.Bar(
                x=sorted_names,
                y=sorted_features,
                marker_color="lightblue",
                text=[f"{val:.3f}" for val in sorted_features],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Molecular Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance",
        xaxis_tickangle=-45,
        height=500,
        width=800,
    )

    return fig


def create_molecular_dashboard(
    df: pd.DataFrame, molecules: List[Chem.Mol]
) -> Dict[str, go.Figure]:
    """
    Create a dashboard with multiple molecular visualizations.

    Args:
        df: DataFrame with molecular properties
        molecules: List of RDKit molecule objects

    Returns:
        Dictionary of plotly figures for dashboard
    """
    dashboard = {}

    # Property distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != "valid"]

    if len(numeric_cols) >= 2:
        dashboard["scatter_matrix"] = plot_molecular_properties(df, numeric_cols[:4])

    # Individual property distributions
    for prop in numeric_cols[:3]:  # First 3 properties
        dashboard[f"{prop}_distribution"] = plot_property_distribution(df, prop)

    # 3D structure for first valid molecule
    valid_molecules = [mol for mol in molecules if mol is not None]
    if valid_molecules:
        dashboard["3d_structure"] = create_3d_conformer_plot(valid_molecules[0])

    return dashboard
