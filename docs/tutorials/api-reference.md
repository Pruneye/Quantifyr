# API Reference

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

#### `extract_atom_features(mol: Mol) -> List[List[float]]`

Extract atomic features for each atom in the molecule.

**Features extracted:**

- Atomic number
- Degree (number of bonds)
- Formal charge
- Hybridization type (SP, SP2, SP3, etc.)
- Aromaticity (boolean)
- Number of implicit hydrogens

**Parameters:**

- `mol` (Mol): RDKit molecule object

**Returns:** List of feature vectors, one per atom

**Example:**

```python
mol = parse_smiles("CCO")
atom_features = extract_atom_features(mol)
print(f"Atom features shape: {len(atom_features)} atoms x {len(atom_features[0])} features")
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

---

## viz Module

### 2D Visualization

#### `draw_molecule_2d(mol: Mol, width: int = 300, height: int = 300, **kwargs) -> PIL.Image`

Generate 2D molecular structure diagrams.

**Parameters:**

- `mol` (Mol): RDKit molecule object
- `width` (int): Image width in pixels (default: 300)
- `height` (int): Image height in pixels (default: 300)
- `**kwargs`: Additional RDKit drawing options

**Returns:** PIL Image object

**Example:**

```python
from data_utils import parse_smiles
from viz import draw_molecule_2d

mol = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
img = draw_molecule_2d(mol, width=400, height=300)
img.save("caffeine_2d.png")
img.show()
```

### 3D Visualization

#### `create_3d_conformer_plot(mol: Mol, **kwargs) -> plotly.graph_objects.Figure`

Create interactive 3D molecular conformer visualization.

**Features:**

- Rotatable 3D structure
- Atom labels and element-based coloring
- Bond representations
- Interactive hover information

**Parameters:**

- `mol` (Mol): RDKit molecule object
- `**kwargs`: Additional plotting options

**Returns:** Plotly Figure object

**Example:**

```python
mol = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
fig = create_3d_conformer_plot(mol)
fig.show()  # Opens in browser
fig.write_html("caffeine_3d.html")
```

### Property Analysis

#### `plot_molecular_properties(df: pd.DataFrame, **kwargs) -> plotly.graph_objects.Figure`

Create interactive scatter matrix of molecular properties.

**Features:**

- Scatter plots for all property combinations
- Hover information showing SMILES and values
- Configurable markers and colors

**Parameters:**

- `df` (pd.DataFrame): Molecular DataFrame from `create_molecular_dataframe()`
- `**kwargs`: Additional plotting options

**Returns:** Plotly Figure object

**Example:**

```python
from data_utils import create_molecular_dataframe
from viz import plot_molecular_properties

smiles_list = ["CCO", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
df = create_molecular_dataframe(smiles_list)
fig = plot_molecular_properties(df)
fig.show()
```

#### `plot_property_distribution(df: pd.DataFrame, properties: List[str], **kwargs) -> plotly.graph_objects.Figure`

Analyze property distributions with histograms and box plots.

**Parameters:**

- `df` (pd.DataFrame): Molecular DataFrame
- `properties` (List[str]): List of property names to analyze
- `**kwargs`: Additional plotting options

**Returns:** Multi-subplot figure with distribution plots

#### `create_molecular_dashboard(df: pd.DataFrame) -> plotly.graph_objects.Figure`

Generate comprehensive molecular analysis dashboard.

**Features:**

- Property scatter plots
- Distribution histograms
- Statistical summaries
- Interactive filtering capabilities

**Parameters:**

- `df` (pd.DataFrame): Molecular DataFrame

**Returns:** Plotly Figure object with multiple subplots

**Example:**

```python
# Analyze diverse molecule set
smiles = [
    "CCO", "CCC", "CCCC",                    # alkanes
    "c1ccccc1", "c1ccc2ccccc2c1",            # aromatics
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",         # caffeine
    "CC(=O)OC1=CC=CC=C1C(=O)O"               # aspirin
]

df = create_molecular_dataframe(smiles)
dashboard = create_molecular_dashboard(df)
dashboard.show()
```

### Network Visualization

#### `plot_molecular_network(mol: Mol, **kwargs) -> plotly.graph_objects.Figure`

Visualize molecular graph structure as an interactive network.

**Features:**

- Nodes represent atoms (colored by element)
- Edges represent bonds (sized by bond order)
- Interactive layout with hover information
- Force-directed positioning

**Parameters:**

- `mol` (Mol): RDKit molecule object
- `**kwargs`: Additional plotting options

**Returns:** Plotly Figure object

**Example:**

```python
mol = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
fig = plot_molecular_network(mol)
fig.show()
```

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
