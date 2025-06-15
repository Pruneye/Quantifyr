"""
3D molecular structure visualization.

This module provides interactive 3D molecular plotting with smart zoom,
theme support, and clean controls.
"""

from typing import Optional
import numpy as np
import plotly.graph_objects as go
from rdkit import Chem
from .themes import get_current_theme, get_contrasting_text_color
from .colors import ELEMENT_COLORS, get_element_symbol
from .core import apply_quantifyr_theme


def create_3d_conformer_plot(
    mol: Chem.Mol, molecule_name: Optional[str] = None, smiles: Optional[str] = None
) -> go.Figure:
    """
    Create beautiful 3D visualization of molecular conformer with smart zoom and titles.

    Args:
        mol: RDKit molecule object with 3D coordinates
        molecule_name: Optional molecule name for title
        smiles: Optional SMILES string for title

    Returns:
        Clean 3D molecular structure plot with Quantifyr theme and optimal zoom
    """
    if mol is None:
        raise ValueError("Cannot plot None molecule")

    # Check if molecule has 3D coordinates
    if mol.GetNumConformers() == 0:
        # Generate 3D conformer if not present
        try:
            from rdkit.Chem import rdDistGeom

            rdDistGeom.EmbedMolecule(mol)
        except Exception as e:
            raise ValueError(f"Failed to generate 3D coordinates: {e}")

    try:
        conf = mol.GetConformer()
    except Exception as e:
        raise ValueError(f"Cannot access molecular conformer: {e}")

    # Extract atomic positions and elements
    positions = []
    atom_info = []

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        atomic_num = atom.GetAtomicNum()
        element_symbol = get_element_symbol(atomic_num)
        atom_info.append((atomic_num, element_symbol))

    positions = np.array(positions)

    # Calculate molecule bounds for smart zoom (MUCH CLOSER like 2D)
    if len(positions) > 0:
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)
        center = (pos_min + pos_max) / 2
        size = np.max(pos_max - pos_min)
        # Much closer zoom like 2D - molecules should fill the view nicely
        camera_distance = max(0.8, min(1.8, size * 0.6 + 0.5))
    else:
        center = np.array([0, 0, 0])
        camera_distance = 0.8

    # Create 3D scatter plot with element symbols and better text contrast
    colors = [ELEMENT_COLORS.get(info[0], "#FFB6C1") for info in atom_info]
    text_labels = [info[1] for info in atom_info]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers+text",
                marker=dict(
                    size=20,  # Larger atoms like 2D
                    color=colors,
                    line=dict(width=2, color="white"),
                    opacity=0.9,
                ),
                text=text_labels,
                textfont=dict(
                    size=16,  # Larger text like 2D
                    color=[get_contrasting_text_color(color) for color in colors],
                    # Smart contrast
                    family=get_current_theme()["font_family"],
                ),
                textposition="middle center",
                hovertemplate="<b>%{text} Element</b><br>3D Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
                showlegend=False,
            )
        ]
    )

    # Add bonds as clean lines
    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()

        pos1 = positions[atom1_idx]
        pos2 = positions[atom2_idx]

        fig.add_trace(
            go.Scatter3d(
                x=[pos1[0], pos2[0], None],
                y=[pos1[1], pos2[1], None],
                z=[pos1[2], pos2[2], None],
                mode="lines",
                line=dict(color=get_current_theme()["secondary_color"], width=6),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Generate smart title
    if molecule_name:
        title = f"3D Structure: {molecule_name}"
    elif smiles:
        title = f"3D Structure: {smiles[:30]}{'...' if len(smiles) > 30 else ''}"
    else:
        title = "3D Molecular Structure"

    # Clean, minimal 3D scene with smart zoom and clean controls
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="cube",
            camera=dict(
                eye=dict(x=camera_distance, y=camera_distance, z=camera_distance),
                center=dict(x=center[0], y=center[1], z=center[2]),
            ),
            # Hide all axis elements for clean look
            xaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                showaxeslabels=False,
                title="",
                visible=False,
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                showaxeslabels=False,
                title="",
                visible=False,
            ),
            zaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                showaxeslabels=False,
                title="",
                visible=False,
            ),
            bgcolor="rgba(0,0,0,0)",  # Transparent background
        ),
        width=900,
        height=700,
        showlegend=False,
        # Custom clean toolbar for 3D
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color=get_current_theme()["font_color"],
            activecolor=get_current_theme()["accent_color"],
            orientation="v",
            remove=[
                "select2d",
                "lasso2d",
                "autoScale2d",
                "resetScale2d",
                "pan2d",
                "zoom2d",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
                "toggleSpikelines",
                "hoverClosest3d",
            ],
            add=[
                "zoom3d",
                "pan3d",
                "orbitRotation",
                "tableRotation",
                "resetCameraDefault3d",
            ],
        ),
    )

    return apply_quantifyr_theme(fig)
