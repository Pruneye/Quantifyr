# API Reference

Complete reference for Quantifyr's data processing and dual-theme visualization capabilities.

## data_utils Module

### Molecular Parsing

#### `parse_smiles(smiles: str, config: Optional[MoleculeConfig] = None) -> Mol`

Parse SMILES string into RDKit molecule object with comprehensive error handling.

**Parameters:**

- `smiles` (str): SMILES string representation
- `config` (MoleculeConfig, optional): Configuration for processing settings

**Returns:** RDKit Mol object

**Raises:** ValueError if SMILES string is invalid

**Example:**

```python
from data_utils import parse_smiles, MoleculeConfig

# Basic parsing
mol = parse_smiles("CCO")

# Custom configuration
config = MoleculeConfig(add_hydrogens=True, max_atoms=100)
mol = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", config)
```

#### `MoleculeConfig`

Configuration class for molecular processing options.

**Attributes:**

- `add_hydrogens` (bool): Whether to add explicit hydrogens (default: False)
- `sanitize` (bool): Whether to sanitize molecules (default: True)
- `max_atoms` (int): Maximum number of atoms allowed (default: 200)

### Feature Extraction

#### `extract_atom_features(mol: Mol) -> np.ndarray`

Extract comprehensive atomic features from molecule for ML models.

**Feature vector (6 features per atom):**

1. **Atomic number** - Element identity (C=6, N=7, O=8, etc.)
2. **Degree** - Number of bonded neighbors
3. **Formal charge** - Charge on atom
4. **Hybridization** - SP/SP2/SP3 hybridization state
5. **Aromaticity** - Is atom in aromatic ring (1/0)
6. **Hydrogen count** - Number of attached hydrogens

**Parameters:**

- `mol` (Mol): RDKit molecule object (from parse_smiles)

**Returns:** Array of shape (n_atoms, 6) with atomic features

**Example:**

```python
mol = parse_smiles("CCO")  # ethanol
features = extract_atom_features(mol)
print(f"Shape: {features.shape}")  # (9, 6) - 9 atoms, 6 features
# First atom (Carbon): [6.0, 4.0, 0.0, 3.0, 0.0, 3.0]
# Element C, degree 4, no charge, SP3, not aromatic, 3 H atoms
```

#### `extract_bond_features(mol: Mol) -> List[List[float]]`

Extract bond features for each bond in the molecule.

**Features extracted:**

- Bond type (single=1, double=2, triple=3, aromatic=1.5)
- Conjugation status (boolean)
- Ring membership (boolean)

**Parameters:**

- `mol` (Mol): RDKit molecule object

**Returns:** List of feature vectors, one per bond

### Graph Construction

#### `mol_to_graph(mol: Mol, include_features: bool = True) -> Data`

Convert RDKit molecule to PyTorch Geometric graph representation.

**Parameters:**

- `mol` (Mol): RDKit molecule object
- `include_features` (bool): Whether to include node/edge features (default: True)

**Returns:** PyTorch Geometric Data object with:

- `x`: Node features (if include_features=True)
- `edge_index`: Edge connectivity in COO format
- `edge_attr`: Edge features (if include_features=True)

**Example:**

```python
mol = parse_smiles("CCO")
graph = mol_to_graph(mol)

print(f"Nodes: {graph.num_nodes}")
print(f"Edges: {graph.num_edges}")
print(f"Node features: {graph.x.shape}")
print(f"Edge features: {graph.edge_attr.shape}")
```

### Property Calculation

#### `compute_molecular_descriptors(mol: Mol) -> Dict[str, float]`

Calculate comprehensive molecular properties and descriptors.

**Properties calculated:**

- **MW**: Molecular weight (Da)
- **LogP**: Partition coefficient (lipophilicity)
- **TPSA**: Topological polar surface area (Ų)
- **NumRotatableBonds**: Number of rotatable bonds
- **NumHBD**: Hydrogen bond donors
- **NumHBA**: Hydrogen bond acceptors
- **NumRings**: Number of rings

**Parameters:**

- `mol` (Mol): RDKit molecule object

**Returns:** Dictionary mapping property names to values

**Example:**

```python
mol = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # caffeine
props = compute_molecular_descriptors(mol)

print(f"Molecular Weight: {props['MW']:.2f} Da")
print(f"LogP: {props['LogP']:.2f}")
print(f"TPSA: {props['TPSA']:.2f} Ų")
```

### Batch Processing

#### `load_molecule_dataset(smiles_list: List[str], labels: Optional[List] = None) -> List[Mol]`

Load and validate multiple SMILES strings into molecule objects.

**Parameters:**

- `smiles_list` (List[str]): List of SMILES strings
- `labels` (List, optional): Optional labels/targets for molecules

**Returns:** List of RDKit Mol objects (None for invalid SMILES)

#### `create_molecular_dataframe(smiles_list: List[str]) -> pd.DataFrame`

Create comprehensive molecular analysis DataFrame with computed properties.

**Parameters:**

- `smiles_list` (List[str]): List of SMILES strings

**Returns:** Pandas DataFrame with columns:

- `SMILES`: Original SMILES string
- `Valid`: Whether parsing succeeded
- `MW`, `LogP`, `TPSA`, etc.: Computed molecular properties

**Example:**

```python
smiles = ["CCO", "c1ccccc1", "INVALID", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
df = create_molecular_dataframe(smiles)

# Filter valid molecules
valid_df = df[df['Valid']]
print(f"Valid molecules: {len(valid_df)}/{len(df)}")
print(valid_df[['SMILES', 'MW', 'LogP', 'TPSA']].head())
```

### Utility Functions

#### `get_element_symbol(atomic_num: int) -> str`

Get element symbol from atomic number using RDKit's periodic table.

**Parameters:**

- `atomic_num` (int): Atomic number (1 for H, 6 for C, etc.)

**Returns:** Element symbol string (e.g., "H", "C", "N", "O")

**Example:**

```python
from data_utils import get_element_symbol

print(get_element_symbol(6))   # "C"
print(get_element_symbol(7))   # "N" 
print(get_element_symbol(8))   # "O"
print(get_element_symbol(999)) # "#999" (unknown elements)
```

---

## viz Module

The visualization module provides publication-ready plots with dual-theme support and smart features. All plots automatically apply your chosen theme with intelligent styling.

### Smart Dual Theming System

#### `set_theme(theme: str) -> None`

Set the global visualization theme affecting all subsequent plots.

**Parameters:**

- `theme` (str): Theme name - "dark" or "light"

**Example:**

```python
from viz import set_theme

set_theme('dark')   # Dark theme with midnight background
set_theme('light')  # Light theme for presentations
```

#### `get_current_theme() -> Dict`

Get the current theme configuration dictionary.

### Theme Features

All visualization functions automatically apply your chosen theme featuring:

- **Dual theme support** - Dark midnight or clean light backgrounds
- **Smart text contrast** - Dark text on light atoms, light text on dark atoms  
- **Professional color palettes** with theme-matched accents
- **Element-specific CPK colors** for all atoms
- **Clean typography** with Inter font family
- **Meaningful titles** using molecule names when provided

### 2D Visualization

#### `draw_molecule_2d(mol: Mol, size: Tuple[int, int] = (400, 400)) -> plotly.graph_objects.Figure`

Generate interactive 2D molecular structure plots with theme support.

**NEW Features:**

- **Interactive plots** instead of static images  
- **Theme-matched styling** with background and bond colors
- **Smart text contrast** on all atom colors
- **CPK element coloring** with element symbols
- **Hover information** showing element type and coordinates
- **Different bond styles** for single/double/triple bonds

**Parameters:**

- `mol` (Mol): RDKit molecule object
- `size` (Tuple[int, int]): Figure size (width, height) - default: (400, 400)

**Returns:** Plotly Figure object with Quantifyr theme applied

**Example:**

```python
from data_utils import parse_smiles
from viz import draw_molecule_2d, set_theme

set_theme('dark')  # or 'light'
mol = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
fig = draw_molecule_2d(mol, size=(500, 400))
fig.show()  # Interactive plot with theme
fig.write_html("caffeine_2d.html")
```

### 3D Visualization

#### `create_3d_conformer_plot(mol: Mol, molecule_name: Optional[str] = None, smiles: Optional[str] = None) -> plotly.graph_objects.Figure`

Create smart 3D molecular conformer visualization with perfect zoom and intelligent features.

**NEW Smart Features:**

- **Perfect zoom level** automatically calculated based on molecule size
- **Smart text contrast** - Dark text on light atoms, light text on dark atoms  
- **Meaningful titles** using molecule names or SMILES instead of generic titles
- **Custom clean controls** with only essential 3D navigation tools
- **Theme-adaptive colors** for both dark and light modes
- **Clean 3D space** without axis clutter
- **Enhanced hover information** with molecule context

**Parameters:**

- `mol` (Mol): RDKit molecule object
- `molecule_name` (str, optional): Molecule name for smart title
- `smiles` (str, optional): SMILES string for title if no name provided

**Returns:** Plotly Figure object with optimal zoom and Quantifyr theme

**Example:**

```python
mol = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # caffeine
fig = create_3d_conformer_plot(
    mol, 
    molecule_name="Caffeine",
    smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
)
fig.show()  # Opens at perfect zoom level with meaningful title!
```

### Property Analysis

#### `plot_molecular_properties(df: pd.DataFrame, properties: List[str]) -> plotly.graph_objects.Figure`

Create interactive scatter matrix of molecular properties with professional styling.

**Features:**

- **Quantifyr theme** applied automatically
- **Professional color scheme** with cyan accents
- **Hover information** showing SMILES and values
- **Clean scatter plots** for all property combinations

**Parameters:**

- `df` (pd.DataFrame): Molecular DataFrame from `create_molecular_dataframe()`
- `properties` (List[str]): List of property names to plot

**Returns:** Plotly Figure object with consistent theming

**Example:**

```python
from data_utils import create_molecular_dataframe
from viz import plot_molecular_properties

smiles_list = ["CCO", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
df = create_molecular_dataframe(smiles_list)
fig = plot_molecular_properties(df, ["molecular_weight", "logp", "tpsa"])
fig.show()
```

#### `plot_property_distribution(df: pd.DataFrame, property_name: str) -> plotly.graph_objects.Figure`

Analyze individual property distributions with professional styling.

**Features:**

- **Dual-plot layout** with histogram and box plot
- **Quantifyr theme** with cyan accents
- **Professional typography** and spacing

**Parameters:**

- `df` (pd.DataFrame): Molecular DataFrame
- `property_name` (str): Single property name to analyze

**Returns:** Multi-subplot figure with distribution analysis

### Network Visualization

#### `plot_molecular_network(graphs: List[Data], max_molecules: int = 10, molecule_names: Optional[List[str]] = None, smiles_list: Optional[List[str]] = None) -> plotly.graph_objects.Figure`

Visualize molecular graphs as smart network plots with meaningful titles and enhanced readability.

**NEW Smart Features:**

- **Meaningful subplot titles** using molecule names instead of "Molecule 1, 2, 3..."
- **Smart text contrast** - Dark text on light atoms, light text on dark atoms
- **Larger atoms** with better visibility (22px)
- **3-column layout** for optimal readability  
- **Theme-adaptive edge colors** and styling
- **Enhanced hover information** with molecule context
- **Clean controls** with simplified toolbar

**Parameters:**

- `graphs` (List[Data]): List of PyTorch Geometric Data objects
- `max_molecules` (int): Maximum number of molecules to display (default: 10)
- `molecule_names` (List[str], optional): List of molecule names for titles
- `smiles_list` (List[str], optional): List of SMILES strings for titles

**Returns:** Plotly Figure object with smart network visualization

**Example:**

```python
from data_utils import load_molecule_dataset

smiles = ["CCO", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)OC1=CC=CC=C1C(=O)O"]
molecule_names = ["Ethanol", "Benzene", "Caffeine", "Aspirin"]
graphs, _ = load_molecule_dataset(smiles)

fig = plot_molecular_network(
    graphs, 
    max_molecules=4,
    molecule_names=molecule_names,  # Meaningful titles!
    smiles_list=smiles
)
fig.show()  # Shows molecules with readable titles and element symbols
```

### Feature Analysis

#### `plot_feature_importance(features: np.ndarray, feature_names: List[str]) -> plotly.graph_objects.Figure`

Plot feature importance analysis with professional styling.

**Features:**

- **Gradient coloring** highlighting top features
- **Quantifyr theme** with consistent colors
- **Clean bar chart** with value labels

**Parameters:**

- `features` (np.ndarray): Feature importance values
- `feature_names` (List[str]): Names of features

**Returns:** Plotly Figure object with feature importance visualization

## Removed Functions

### ~~`create_molecular_dashboard()`~~ **[REMOVED]**

This function has been removed for better flexibility. Instead, use individual plotting functions:

- `plot_molecular_properties()` for scatter matrices
- `plot_property_distribution()` for individual property analysis  
- `create_3d_conformer_plot()` for 3D structures
- `plot_molecular_network()` for graph networks

This approach gives you more control over customization and layout.

---

## Error Handling

All functions include comprehensive error handling:

- **Invalid SMILES**: Functions return None or raise ValueError with descriptive messages
- **Empty molecules**: Gracefully handled with appropriate warnings
- **Feature extraction errors**: Fallback to default values where possible
- **Visualization errors**: Return empty figures with error messages

## Type Hints

All functions include complete type annotations for better IDE support and code clarity.

## Performance Notes

- **Batch processing**: Use `create_molecular_dataframe()` for multiple molecules
- **Large molecules**: Set appropriate limits in `MoleculeConfig`
- **Memory usage**: 3D conformer generation can be memory-intensive for large molecules
- **Visualization**: Interactive plots may be slow for very large datasets
