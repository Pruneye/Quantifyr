# Data Processing Tutorial

Learn how to process molecular data in Quantifyr with comprehensive SMILES parsing, feature extraction, and batch analysis. All functions provide clear, human-readable output with element symbols, detailed molecular information, and modern theme support.

## Basic SMILES Parsing

### Simple Parsing

```python
from data_utils import parse_smiles

# Parse basic molecules
ethanol = parse_smiles("CCO")
benzene = parse_smiles("c1ccccc1")
caffeine = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

print(f"Ethanol atoms: {ethanol.GetNumAtoms()}")
print(f"Benzene atoms: {benzene.GetNumAtoms()}")
print(f"Caffeine atoms: {caffeine.GetNumAtoms()}")
```

### Configuration Options

```python
from data_utils import parse_smiles, MoleculeConfig

# Custom configuration
config = MoleculeConfig(
    add_hydrogens=True,    # Add explicit hydrogens
    sanitize=True,         # Sanitize molecules
    max_atoms=100         # Limit molecule size
)

mol = parse_smiles("CCO", config)
print(f"With hydrogens: {mol.GetNumAtoms()} atoms")
```

### Error Handling

```python
# Invalid SMILES handling
try:
    invalid_mol = parse_smiles("INVALID_SMILES")
except ValueError as e:
    print(f"Error: {e}")

# Check if molecule is valid
mol = parse_smiles("CCO")
if mol is not None:
    print("Valid molecule")
```

## Feature Extraction

### Atomic Features with Element Symbols

```python
from data_utils import parse_smiles, extract_atom_features, get_element_symbol

mol = parse_smiles("CCO")
atom_features = extract_atom_features(mol)

print(f"Number of atoms: {len(atom_features)}")
print(f"Features per atom: {len(atom_features[0])}")

# Show first atom with element symbol
first_atom = atom_features[0]
element = get_element_symbol(int(first_atom[0]))
print(f"First atom ({element}): {first_atom}")
```

**Feature meanings (6 features per atom):**

1. **Atomic number** - Element identity (C=6, N=7, O=8, etc.)
2. **Degree** - Number of bonded neighbors
3. **Formal charge** - Charge on atom  
4. **Hybridization** - SP/SP2/SP3 hybridization state
5. **Aromaticity** - Is atom in aromatic ring (1/0)
6. **Hydrogen count** - Number of attached hydrogens

**Understanding the output:**

- First atom (Carbon): `[6.0, 4.0, 0.0, 3.0, 0.0, 3.0]`
- Element C, degree 4, no charge, SP3, not aromatic, 3 H atoms

### Bond Features

```python
from data_utils import extract_bond_features

mol = parse_smiles("c1ccccc1")  # benzene
bond_features = extract_bond_features(mol)

print(f"Number of bonds: {len(bond_features)}")
print(f"Features per bond: {len(bond_features[0])}")
print(f"First bond features: {bond_features[0]}")
```

**Feature meanings (3 features per bond):**

1. **Bond type** - Single=1, Double=2, Triple=3, Aromatic=1.5
2. **Conjugation** - Is bond part of conjugated system (1/0)
3. **Ring membership** - Is bond in a ring (1/0)

**Understanding the output:**

- Benzene bond: `[1.5, 1.0, 1.0]` - Aromatic bond, conjugated, in ring

## Graph Construction

### Basic Graph Creation

```python
from data_utils import parse_smiles, mol_to_graph

mol = parse_smiles("CCO")
graph = mol_to_graph(mol)

print(f"Graph nodes: {graph.num_nodes}")
print(f"Graph edges: {graph.num_edges}")
print(f"Node features shape: {graph.x.shape}")
print(f"Edge features shape: {graph.edge_attr.shape}")
print(f"Edge connectivity shape: {graph.edge_index.shape}")
```

### Graph Without Features

```python
# Create graph without features (topology only)
graph_no_features = mol_to_graph(mol, include_features=False)
print(f"Has node features: {hasattr(graph_no_features, 'x')}")
print(f"Has edge features: {hasattr(graph_no_features, 'edge_attr')}")
```

### Understanding Edge Index

```python
import torch

mol = parse_smiles("CCO")
graph = mol_to_graph(mol)

# Edge index is in COO format: [2, num_edges]
edge_index = graph.edge_index
print(f"Edge index shape: {edge_index.shape}")
print(f"Edge connections:\n{edge_index}")

# Each column represents an edge: [source_node, target_node]
for i in range(edge_index.shape[1]):
    src, dst = edge_index[:, i]
    print(f"Edge {i}: atom {src} -> atom {dst}")
```

## Property Calculation

### Individual Properties

```python
from data_utils import parse_smiles, compute_molecular_descriptors

molecules = {
    "ethanol": "CCO",
    "benzene": "c1ccccc1", 
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O"
}

for name, smiles in molecules.items():
    mol = parse_smiles(smiles)
    props = compute_molecular_descriptors(mol)
    
    print(f"\n{name.upper()}:")
    print(f"  Molecular Weight: {props['MW']:.2f} Da")
    print(f"  LogP: {props['LogP']:.2f}")
    print(f"  TPSA: {props['TPSA']:.2f} Ų")
    print(f"  Rotatable Bonds: {props['NumRotatableBonds']}")
    print(f"  H-Bond Donors: {props['NumHBD']}")
    print(f"  H-Bond Acceptors: {props['NumHBA']}")
```

### Property Interpretation

**Drug-like descriptors for QSAR analysis:**

- **molecular_weight** - Molecular weight in Daltons (46.1 for ethanol)
- **logp** - Lipophilicity/partition coefficient (-0.31 for ethanol = hydrophilic)  
- **tpsa** - Topological polar surface area in Ų (affects membrane permeability)
- **num_rotatable_bonds** - Flexibility measure (more = more flexible)
- **num_hbd** - Hydrogen bond donors (NH, OH groups)
- **num_hba** - Hydrogen bond acceptors (N, O atoms)
- **num_rings** - Ring count (important for rigidity)
- **num_aromatic_rings** - Aromatic ring count

## Batch Processing

### Loading Multiple Molecules

```python
from data_utils import load_molecule_dataset

smiles_list = [
    "CCO",           # ethanol
    "c1ccccc1",      # benzene
    "INVALID",       # invalid SMILES
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine
]

molecules = load_molecule_dataset(smiles_list)
print(f"Loaded {len([m for m in molecules if m is not None])}/{len(molecules)} valid molecules")

# Process valid molecules
for i, mol in enumerate(molecules):
    if mol is not None:
        print(f"Molecule {i}: {mol.GetNumAtoms()} atoms")
    else:
        print(f"Molecule {i}: Invalid")
```

### Creating Analysis DataFrames

```python
from data_utils import create_molecular_dataframe
import pandas as pd

# Diverse molecule set
smiles_list = [
    "CCO",                                    # ethanol
    "CCC",                                    # propane
    "c1ccccc1",                              # benzene
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",         # caffeine
    "CC(=O)OC1=CC=CC=C1C(=O)O",              # aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",        # ibuprofen
    "INVALID_SMILES"                          # invalid
]

df = create_molecular_dataframe(smiles_list)

# Display results
print("Molecular Analysis Results:")
print(df.to_string(index=False))

# Filter valid molecules
valid_df = df[df['Valid']].copy()
print(f"\nValid molecules: {len(valid_df)}/{len(df)}")

# Statistical summary
print("\nProperty Statistics:")
numeric_cols = ['MW', 'LogP', 'TPSA', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumRings']
print(valid_df[numeric_cols].describe())
```

### Advanced Filtering

```python
# Filter by properties
drug_like = valid_df[
    (valid_df['MW'] <= 500) &           # Lipinski's rule
    (valid_df['LogP'] <= 5) &           # Lipophilicity
    (valid_df['NumHBD'] <= 5) &         # H-bond donors
    (valid_df['NumHBA'] <= 10)          # H-bond acceptors
]

print(f"Drug-like molecules: {len(drug_like)}/{len(valid_df)}")
print(drug_like[['SMILES', 'MW', 'LogP', 'NumHBD', 'NumHBA']])
```

## Working with Large Datasets

### Memory-Efficient Processing

```python
def process_large_dataset(smiles_list, batch_size=1000):
    """Process large SMILES datasets in batches."""
    results = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        batch_df = create_molecular_dataframe(batch)
        results.append(batch_df)
        print(f"Processed batch {i//batch_size + 1}: {len(batch)} molecules")
    
    return pd.concat(results, ignore_index=True)

# Example with simulated large dataset
large_smiles = ["CCO", "c1ccccc1", "CCC"] * 100  # 300 molecules
result_df = process_large_dataset(large_smiles, batch_size=50)
print(f"Total processed: {len(result_df)} molecules")
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def process_smiles_chunk(smiles_chunk):
    """Process a chunk of SMILES strings."""
    return create_molecular_dataframe(smiles_chunk)

def parallel_processing(smiles_list, n_workers=None):
    """Process SMILES list using multiple processes."""
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Split into chunks
    chunk_size = len(smiles_list) // n_workers
    chunks = [smiles_list[i:i+chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_smiles_chunk, chunks))
    
    return pd.concat(results, ignore_index=True)

# Example usage (commented out to avoid multiprocessing issues in docs)
# result_df = parallel_processing(large_smiles, n_workers=4)
```

## Best Practices

### Using Error Handling

```python
def safe_molecular_processing(smiles_list):
    """Safely process molecules with comprehensive error handling."""
    results = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            mol = parse_smiles(smiles)
            if mol is not None:
                props = compute_molecular_descriptors(mol)
                graph = mol_to_graph(mol)
                
                results.append({
                    'index': i,
                    'smiles': smiles,
                    'valid': True,
                    'atoms': mol.GetNumAtoms(),
                    'bonds': mol.GetNumBonds(),
                    'graph_nodes': graph.num_nodes,
                    'graph_edges': graph.num_edges,
                    **props
                })
            else:
                results.append({
                    'index': i,
                    'smiles': smiles,
                    'valid': False,
                    'error': 'Invalid molecule'
                })
                
        except Exception as e:
            results.append({
                'index': i,
                'smiles': smiles,
                'valid': False,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

# Test with mixed valid/invalid SMILES
test_smiles = ["CCO", "INVALID", "c1ccccc1", ""]
result_df = safe_molecular_processing(test_smiles)
print(result_df)
```

### Performance Tips

1. **Use batch processing** for multiple molecules
2. **Cache results** for repeated calculations
3. **Set molecule size limits** to avoid memory issues
4. **Use parallel processing** for large datasets
5. **Validate SMILES** before expensive operations

### Memory Management

```python
import gc

def memory_efficient_processing(smiles_list):
    """Process molecules with explicit memory management."""
    results = []
    
    for i, smiles in enumerate(smiles_list):
        mol = parse_smiles(smiles)
        if mol is not None:
            # Process immediately and store only results
            props = compute_molecular_descriptors(mol)
            results.append(props)
            
            # Clear molecule from memory
            del mol
            
            # Periodic garbage collection
            if i % 100 == 0:
                gc.collect()
    
    return results
```

This tutorial covers the core data processing capabilities. Next, explore the [Visualization Tutorial](visualization.md) to learn about creating molecular visualizations.
