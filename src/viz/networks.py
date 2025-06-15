"""
Molecular network visualization.

This module provides network-style visualization of molecular graphs
with smart titles and enhanced readability.
"""

from typing import List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from torch_geometric.data import Data
from .themes import get_current_theme, get_contrasting_text_color
from .colors import ELEMENT_COLORS, get_element_symbol
from .core import apply_quantifyr_theme


def plot_molecular_network(
    graphs: List[Data],
    max_molecules: int = 10,
    molecule_names: Optional[List[str]] = None,
    smiles_list: Optional[List[str]] = None,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
) -> go.Figure:
    """
    Visualize molecular graphs as network plots with element symbols and smart titles.

    Args:
        graphs: List of PyTorch Geometric Data objects
        max_molecules: Maximum number of molecules to display
        molecule_names: Optional list of molecule names for titles
        smiles_list: Optional list of SMILES strings for titles
        cols: Optional number of columns (default: auto-calculated)
        rows: Optional number of rows (default: auto-calculated)

    Returns:
        Plotly figure with molecular networks and Quantifyr theme
    """
    if not graphs:
        raise ValueError("No graphs provided")

    # Limit number of molecules for readability
    graphs = graphs[:max_molecules]
    n_graphs = len(graphs)

    # Generate meaningful titles
    titles = []
    for i in range(n_graphs):
        if molecule_names and i < len(molecule_names):
            titles.append(molecule_names[i])
        elif smiles_list and i < len(smiles_list):
            # Truncate long SMILES for readability
            smiles = smiles_list[i]
            title = smiles if len(smiles) <= 20 else smiles[:20] + "..."
            titles.append(title)
        else:
            titles.append(f"Molecule {i + 1}")

    # Calculate grid dimensions (allow user customization)
    if cols is None:
        cols = min(3, n_graphs)  # Default to 3 for better visibility
    if rows is None:
        rows = (n_graphs + cols - 1) // cols

    # Ensure we don't exceed the number of graphs
    total_slots = rows * cols
    if total_slots < n_graphs:
        rows = (n_graphs + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        specs=[[{"type": "scatter"}] * cols for _ in range(rows)],
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    for idx, graph in enumerate(graphs):
        row = idx // cols + 1
        col = idx % cols + 1

        # Validate graph structure
        if not hasattr(graph, "x") or graph.x.shape[0] == 0:
            continue  # Skip empty graphs

        if not hasattr(graph, "edge_index"):
            continue  # Skip graphs without edges

        # Convert to NetworkX
        G = nx.Graph()

        # Add nodes with features
        node_info = []
        for i in range(graph.x.shape[0]):
            try:
                # First feature is atomic number
                atomic_num = int(graph.x[i, 0])
                element_symbol = get_element_symbol(atomic_num)
                G.add_node(i, atomic_num=atomic_num, element=element_symbol)
                node_info.append((atomic_num, element_symbol))
            except (IndexError, ValueError):
                # Skip invalid nodes
                continue

        # Add edges
        edge_list = graph.edge_index.t().numpy()
        G.add_edges_from(edge_list)

        # Layout with better spacing
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

        # Node trace with element colors
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_colors = [ELEMENT_COLORS.get(info[0], "#FFB6C1") for info in node_info]
        node_text = [info[1] for info in node_info]

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
                line=dict(width=2, color=get_current_theme()["secondary_color"]),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Add node trace with smart text contrast
        text_colors = [get_contrasting_text_color(color) for color in node_colors]
        theme = get_current_theme()

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=22,  # Slightly larger for better visibility
                    color=node_colors,
                    line=dict(width=2, color=theme["secondary_color"]),
                ),
                text=node_text,
                textfont=dict(
                    size=11,
                    color=text_colors,  # Smart contrast based on atom color
                    family=theme["font_family"],
                ),
                textposition="middle center",
                showlegend=False,
                hovertemplate="<b>%{text} Element</b><br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>",
            ),
            row=row,
            col=col,
        )

    # Determine main title based on input
    if molecule_names:
        main_title = f"Molecular Networks: {', '.join(molecule_names[:3])}"
        if len(molecule_names) > 3:
            main_title += f" (and {len(molecule_names) - 3} more)"
    elif smiles_list:
        main_title = f"Molecular Networks: {len(smiles_list)} molecules"
    else:
        main_title = "Molecular Graph Networks"

    fig.update_layout(
        title=main_title,
        height=350 * rows,  # Increased height for better visibility
        width=1000,  # Reduced width for better proportion
        showlegend=False,
        # Clean up the toolbar
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color=get_current_theme()["font_color"],
            activecolor=get_current_theme()["accent_color"],
            orientation="h",
            remove=[
                "select2d",
                "lasso2d",
                "autoScale2d",
                "resetScale2d",
                "toggleSpikelines",
            ],
        ),
    )

    # Remove axes for cleaner look
    fig.update_xaxes(
        showgrid=False, zeroline=False, showticklabels=False, visible=False
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, showticklabels=False, visible=False
    )

    return apply_quantifyr_theme(fig)
