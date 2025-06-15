"""
Property analysis visualization.

This module provides visualization functions for molecular properties,
distributions, and feature importance analysis.
"""

from typing import List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from .themes import get_current_theme
from .core import apply_quantifyr_theme


def plot_molecular_properties(df: pd.DataFrame, properties: List[str]) -> go.Figure:
    """
    Create interactive scatter matrix plot of molecular properties.

    Args:
        df: DataFrame with molecular properties
        properties: List of property names to plot

    Returns:
        Plotly figure with scatter matrix and Quantifyr theme

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

    # Create scatter matrix with theme colors
    theme = get_current_theme()
    fig = px.scatter_matrix(
        valid_df,
        dimensions=properties,
        title="Molecular Properties Analysis",
        hover_data=["smiles"],
        color_discrete_sequence=[theme["accent_color"]],
    )

    fig.update_layout(height=700, width=900, showlegend=False)
    fig.update_traces(
        marker=dict(size=6, opacity=0.7, line=dict(width=0.5, color="white"))
    )

    return apply_quantifyr_theme(fig)


def plot_property_distribution(df: pd.DataFrame, property_name: str) -> go.Figure:
    """
    Plot distribution of molecular property with professional styling.

    Args:
        df: DataFrame with molecular properties
        property_name: Name of property to plot

    Returns:
        Plotly figure with histogram and box plot using Quantifyr theme
    """
    valid_df = df[df["valid"]].copy()

    if valid_df.empty or property_name not in valid_df.columns:
        raise ValueError(f"Property '{property_name}' not found in valid data")

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            f"{
                property_name.title()} Distribution",
            f"{
                property_name.title()} Summary",
        ],
        row_heights=[0.7, 0.3],
        vertical_spacing=0.12,
    )

    # Histogram with theme accent color
    theme = get_current_theme()
    fig.add_trace(
        go.Histogram(
            x=valid_df[property_name],
            nbinsx=25,
            name="Distribution",
            marker_color=theme["accent_color"],
            opacity=0.8,
            marker_line=dict(width=1, color=theme["font_color"]),
        ),
        row=1,
        col=1,
    )

    # Box plot with matching colors (fix color format)
    accent_color = theme["accent_color"].lstrip("#")
    rgba_color = f"rgba({int(accent_color[0:2],
                             16)}, {int(accent_color[2:4],
                                        16)}, {int(accent_color[4:6],
                                                   16)}, 0.3)"

    fig.add_trace(
        go.Box(
            x=valid_df[property_name],
            name="Summary Statistics",
            marker_color=theme["accent_color"],
            line_color=theme["font_color"],
            fillcolor=rgba_color,  # Proper RGBA format
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"Molecular Property Analysis: {property_name.title()}",
        height=550,
        width=850,
        showlegend=False,
    )

    return apply_quantifyr_theme(fig)


def plot_feature_importance(
    features: np.ndarray, feature_names: List[str]
) -> go.Figure:
    """
    Plot feature importance for molecular features with professional styling.

    Args:
        features: Feature importance values
        feature_names: Names of features

    Returns:
        Plotly bar chart with Quantifyr theme
    """
    if len(features) != len(feature_names):
        raise ValueError("Features and feature names must have same length")

    # Sort by importance
    sorted_indices = np.argsort(features)[::-1]
    sorted_features = features[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]

    # Create gradient colors for bars using theme
    theme = get_current_theme()
    colors = [
        theme["accent_color"] if i < 3 else theme["secondary_color"]
        for i in range(len(sorted_features))
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=sorted_names,
                y=sorted_features,
                marker_color=colors,
                marker_line=dict(width=1, color=theme["font_color"]),
                text=[f"{val:.3f}" for val in sorted_features],
                textposition="outside",
                textfont=dict(color=theme["font_color"], size=10),
            )
        ]
    )

    fig.update_layout(
        title="Molecular Feature Importance Analysis",
        xaxis_title="Feature Names",
        yaxis_title="Importance Score",
        xaxis_tickangle=-45,
        height=550,
        width=900,
        yaxis=dict(gridwidth=1),
    )

    return apply_quantifyr_theme(fig)
