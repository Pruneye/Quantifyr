"""
2D molecular structure visualization.

This module provides interactive 2D molecular plotting with theme support,
smart text contrast, and CPK coloring.
"""

from typing import Tuple
import numpy as np
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import rdDepictor
from .themes import get_current_theme, get_contrasting_text_color
from .colors import ELEMENT_COLORS, get_element_symbol
from .core import apply_quantifyr_theme


def draw_molecule_2d(mol: Chem.Mol, size: Tuple[int, int] = (400, 400)) -> go.Figure:
    """
    Draw 2D structure of molecule with Quantifyr theming as interactive plot.

    Args:
        mol: RDKit molecule object
        size: Figure size (width, height)

    Returns:
        Plotly figure with 2D molecular structure matching current theme

    Examples:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> fig = draw_molecule_2d(mol)
        >>> fig.show()
    """
    if mol is None:
        raise ValueError("Cannot draw None molecule")

    # Generate 2D coordinates if not present
    rdDepictor.Compute2DCoords(mol)

    # Get atom positions and info
    conf = mol.GetConformer()
    positions = []
    atom_info = []

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y])
        atomic_num = atom.GetAtomicNum()
        element_symbol = get_element_symbol(atomic_num)
        atom_info.append((atomic_num, element_symbol))

    positions = np.array(positions)

    # Create figure
    fig = go.Figure()

    # Add bonds first (so they appear behind atoms)
    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()

        pos1 = positions[atom1_idx]
        pos2 = positions[atom2_idx]

        # Different line styles for different bond types
        line_width = 3
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            line_width = 5
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            line_width = 7

        theme = get_current_theme()

        fig.add_trace(
            go.Scatter(
                x=[pos1[0], pos2[0], None],
                y=[pos1[1], pos2[1], None],
                mode="lines",
                line=dict(color=theme["secondary_color"], width=line_width),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add atoms
    if len(positions) > 0:
        colors = [ELEMENT_COLORS.get(info[0], "#FFB6C1") for info in atom_info]
        text_labels = [info[1] for info in atom_info]
        text_colors = [get_contrasting_text_color(color) for color in colors]

        fig.add_trace(
            go.Scatter(
                x=positions[:, 0],
                y=positions[:, 1],
                mode="markers+text",
                marker=dict(
                    size=25,
                    color=colors,
                    line=dict(width=2, color=get_current_theme()["secondary_color"]),
                ),
                text=text_labels,
                textfont=dict(
                    size=14,
                    color=text_colors,
                    family=get_current_theme()["font_family"],
                ),
                textposition="middle center",
                showlegend=False,
                hovertemplate="<b>%{text} Element</b><br>2D Position: (%{x:.2f}, %{y:.2f})<extra></extra>",
            )
        )

    # Update layout for clean 2D molecular view
    fig.update_layout(
        title="2D Molecular Structure",
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            visible=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            visible=False,
        ),
        width=size[0],
        height=size[1],
        showlegend=False,
    )

    return apply_quantifyr_theme(fig)
