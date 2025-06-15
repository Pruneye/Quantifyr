"""
Interactive molecular visualization utilities for Quantifyr.

This module provides 2D and 3D visualization functions for molecules,
molecular properties, and graph representations with consistent theming.

This is the main entry point that imports from specialized modules:
- themes: Theme management and smart text contrast
- colors: Element coloring and CPK schemes
- core: Theme application utilities
- molecular_2d: 2D molecular structure visualization
- molecular_3d: 3D molecular structure visualization
- networks: Molecular network visualization
- properties: Property analysis and feature importance
"""

# Import all functionality from specialized modules
from .themes import (
    set_theme,
    get_current_theme,
    get_contrasting_text_color,
)

from .colors import (
    get_element_symbol,
    ELEMENT_COLORS,
)

from .core import (
    apply_quantifyr_theme,
)

from .molecular_2d import (
    draw_molecule_2d,
)

from .molecular_3d import (
    create_3d_conformer_plot,
)

from .networks import (
    plot_molecular_network,
)

from .properties import (
    plot_molecular_properties,
    plot_property_distribution,
    plot_feature_importance,
)

# Re-export all functions for backward compatibility
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
