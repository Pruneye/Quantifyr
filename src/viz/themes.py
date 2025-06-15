"""
Theme management for Quantifyr visualizations.

This module provides dual theme support (dark/light) and smart text contrast
functionality for all visualization components.
"""

from typing import Dict


# Global theme configurations for dark/light mode toggle
QUANTIFYR_THEMES = {
    "dark": {
        "paper_bgcolor": "#0d1117",
        "plot_bgcolor": "#161b22",
        "font_color": "#c9d1d9",
        "grid_color": "#30363d",
        "accent_color": "#00acc1",
        "secondary_color": "#6e7681",
        "font_family": "Inter, sans-serif",
        "font_size": 12,
        "title_font_size": 16,
        "margin": dict(l=60, r=60, t=80, b=60),
    },
    "light": {
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#f6f8fa",
        "font_color": "#24292f",
        "grid_color": "#d0d7de",
        "accent_color": "#0969da",
        "secondary_color": "#656d76",
        "font_family": "Inter, sans-serif",
        "font_size": 12,
        "title_font_size": 16,
        "margin": dict(l=60, r=60, t=80, b=60),
    },
}

# Current theme state (can be toggled)
_CURRENT_THEME = "dark"


def set_theme(theme: str) -> None:
    """Set the global Quantifyr theme (dark or light)."""
    global _CURRENT_THEME
    if theme not in QUANTIFYR_THEMES:
        raise ValueError(f"Theme must be 'dark' or 'light', got '{theme}'")
    _CURRENT_THEME = theme


def get_current_theme() -> Dict:
    """Get the current theme configuration."""
    return QUANTIFYR_THEMES[_CURRENT_THEME]


def get_contrasting_text_color(background_color: str) -> str:
    """Get contrasting text color (black or white) based on background brightness."""
    # Remove # if present
    color = background_color.lstrip("#")

    # Convert to RGB
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)

    # Calculate brightness (luminance)
    brightness = r * 0.299 + g * 0.587 + b * 0.114

    # Return black for light backgrounds, white for dark backgrounds
    return "#000000" if brightness > 128 else "#ffffff"
