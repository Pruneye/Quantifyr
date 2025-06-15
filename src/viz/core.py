"""
Core visualization utilities and theme application.

This module provides the main theme application function and common
utilities used across all visualization components.
"""

import plotly.graph_objects as go
from .themes import get_current_theme


def apply_quantifyr_theme(fig: go.Figure, custom_modebar: bool = False) -> go.Figure:
    """Apply consistent Quantifyr theme to plotly figure with clean controls."""
    theme = get_current_theme()

    modebar_config = dict(
        bgcolor="rgba(0,0,0,0)",
        color=theme["font_color"],
        activecolor=theme["accent_color"],
        remove=["select2d", "lasso2d"] if not custom_modebar else None,
    )

    fig.update_layout(
        paper_bgcolor=theme["paper_bgcolor"],
        plot_bgcolor=theme["plot_bgcolor"],
        font=dict(
            family=theme["font_family"],
            size=theme["font_size"],
            color=theme["font_color"],
        ),
        title_font_size=theme["title_font_size"],
        margin=theme["margin"],
        modebar=(
            modebar_config
            if not fig.layout.modebar or not custom_modebar
            else fig.layout.modebar
        ),
    )

    # Apply consistent, thin axes styling to all axes
    fig.update_xaxes(
        gridcolor=theme["grid_color"],
        linecolor=theme["grid_color"],
        linewidth=1,  # Thin, consistent line width
        gridwidth=0.5,  # Thin grid lines
        zeroline=True,
        zerolinecolor=theme["grid_color"],
        zerolinewidth=1,
    )
    fig.update_yaxes(
        gridcolor=theme["grid_color"],
        linecolor=theme["grid_color"],
        linewidth=1,  # Thin, consistent line width
        gridwidth=0.5,  # Thin grid lines
        zeroline=True,
        zerolinecolor=theme["grid_color"],
        zerolinewidth=1,
    )
    return fig
