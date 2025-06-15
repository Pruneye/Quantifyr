"""
Color schemes and element coloring for molecular visualizations.

This module provides professional CPK coloring for chemical elements
and utility functions for element symbol lookup.
"""

from rdkit import Chem


def get_element_symbol(atomic_num: int) -> str:
    """Get element symbol from atomic number using RDKit."""
    try:
        return Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
    except BaseException:
        return f"#{atomic_num}"


# Professional element colors (CPK coloring with modern hex values)
ELEMENT_COLORS = {
    1: "#FFFFFF",  # H - White
    6: "#2E2E2E",  # C - Dark Gray
    7: "#3050F8",  # N - Blue
    8: "#FF0D0D",  # O - Red
    9: "#90E050",  # F - Green
    15: "#FF8000",  # P - Orange
    16: "#FFFF30",  # S - Yellow
    17: "#1FF01F",  # Cl - Green
    35: "#A62929",  # Br - Dark Red
    53: "#940094",  # I - Purple
    11: "#AB5CF2",  # Na - Violet
    12: "#8AFF00",  # Mg - Light Green
    13: "#BFA6A6",  # Al - Gray
    14: "#F0C8A0",  # Si - Tan
    19: "#8F40D4",  # K - Violet
    20: "#3DFF00",  # Ca - Green
    26: "#E06633",  # Fe - Orange
    30: "#7D80B0",  # Zn - Blue Gray
    25: "#9C7AC7",  # Mn - Purple
    29: "#C88033",  # Cu - Copper
    # Add default for others
    **{
        i: "#FFB6C1"
        for i in range(2, 119)
        if i
        not in [
            1,
            6,
            7,
            8,
            9,
            15,
            16,
            17,
            35,
            53,
            11,
            12,
            13,
            14,
            19,
            20,
            26,
            30,
            25,
            29,
        ]
    },
}
