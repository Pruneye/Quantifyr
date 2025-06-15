"""
Interactive molecular visualization utilities for Quantifyr.

This package provides comprehensive visualization capabilities:
- themes: Theme management and smart text contrast
- colors: Element coloring and CPK schemes
- core: Theme application utilities
- molecular_2d: 2D molecular structure visualization
- molecular_3d: 3D molecular structure visualization
- networks: Molecular network visualization
- properties: Property analysis and feature importance
"""

# Import all functionality from the main module (which imports from submodules)
from .viz import (
    # Theme control
    set_theme,
    get_current_theme,
    get_contrasting_text_color,
    # Colors and elements
    get_element_symbol,
    ELEMENT_COLORS,
    # Core utilities
    apply_quantifyr_theme,
    # 2D visualization
    draw_molecule_2d,
    # 3D visualization
    create_3d_conformer_plot,
    # Network visualization
    plot_molecular_network,
    # Property analysis
    plot_molecular_properties,
    plot_property_distribution,
    plot_feature_importance,
)

__all__ = [
    # Theme control
    "set_theme",
    "get_current_theme",
    "get_contrasting_text_color",
    # Colors and elements
    "get_element_symbol",
    "ELEMENT_COLORS",
    # Core utilities
    "apply_quantifyr_theme",
    # 2D visualization
    "draw_molecule_2d",
    # 3D visualization
    "create_3d_conformer_plot",
    # Network visualization
    "plot_molecular_network",
    # Property analysis
    "plot_molecular_properties",
    "plot_property_distribution",
    "plot_feature_importance",
]
