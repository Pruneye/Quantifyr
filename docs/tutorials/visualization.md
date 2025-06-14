# Visualization Tutorial

This tutorial covers all molecular visualization capabilities in Quantifyr, from basic 2D structures to interactive dashboards.

## 2D Molecular Structures

### Basic 2D Drawing

```python
from data_utils import parse_smiles
from viz import draw_molecule_2d

# Parse molecules
caffeine = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
aspirin = parse_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")

# Create 2D images
caffeine_img = draw_molecule_2d(caffeine)
aspirin_img = draw_molecule_2d(aspirin, width=400, height=300)

# Display images
caffeine_img.show()
aspirin_img.show()

# Save images
caffeine_img.save("caffeine_2d.png")
aspirin_img.save("aspirin_2d.png")
```

### Custom 2D Drawing Options

```python
# High-resolution image
high_res_img = draw_molecule_2d(
    caffeine, 
    width=800, 
    height=600,
    kekulize=True,  # Show explicit double bonds
    wedgeBonds=True  # Show stereochemistry
)

# Multiple molecules in a grid
molecules = [
    ("Ethanol", parse_smiles("CCO")),
    ("Benzene", parse_smiles("c1ccccc1")),
    ("Caffeine", parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")),
    ("Aspirin", parse_smiles("CC(=O)OC1=CC=CC=C1C(=O)O"))
]

for name, mol in molecules:
    img = draw_molecule_2d(mol, width=300, height=200)
    img.save(f"{name.lower()}_2d.png")
    print(f"Saved {name} structure")
```

## 3D Interactive Visualizations

### Basic 3D Conformers

```python
from viz import create_3d_conformer_plot

# Create 3D visualization
mol = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
fig = create_3d_conformer_plot(mol)

# Display in browser
fig.show()

# Save as HTML
fig.write_html("caffeine_3d.html")

# Save as static image
fig.write_image("caffeine_3d.png", width=800, height=600)
```

### Advanced 3D Options

```python
# Custom styling
fig = create_3d_conformer_plot(
    mol,
    atom_size=8,
    bond_width=4,
    show_atom_labels=True,
    background_color='white'
)

# Update layout
fig.update_layout(
    title="Caffeine 3D Structure",
    scene=dict(
        xaxis_title="X (Å)",
        yaxis_title="Y (Å)", 
        zaxis_title="Z (Å)",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )
)

fig.show()
```

### Multiple 3D Conformers

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

molecules = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
}

# Create subplot figure
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
    subplot_titles=list(molecules.keys())
)

for i, (name, smiles) in enumerate(molecules.items(), 1):
    mol = parse_smiles(smiles)
    mol_fig = create_3d_conformer_plot(mol)
    
    # Add traces to subplot
    for trace in mol_fig.data:
        fig.add_trace(trace, row=1, col=i)

fig.update_layout(title="Molecular 3D Structures Comparison")
fig.show()
```

## Property Analysis Visualizations

### Scatter Matrix Plots

```python
from data_utils import create_molecular_dataframe
from viz import plot_molecular_properties

# Create dataset
smiles_list = [
    "CCO",                                    # ethanol
    "CCC",                                    # propane
    "CCCC",                                   # butane
    "c1ccccc1",                              # benzene
    "c1ccc2ccccc2c1",                        # naphthalene
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",         # caffeine
    "CC(=O)OC1=CC=CC=C1C(=O)O",              # aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"         # ibuprofen
]

df = create_molecular_dataframe(smiles_list)
valid_df = df[df['Valid']].copy()

# Create scatter matrix
fig = plot_molecular_properties(valid_df)
fig.show()

# Save interactive plot
fig.write_html("molecular_properties_scatter.html")
```

### Property Distribution Analysis

```python
from viz import plot_property_distribution

# Analyze specific properties
properties = ['MW', 'LogP', 'TPSA', 'NumRotatableBonds']
fig = plot_property_distribution(valid_df, properties)
fig.show()

# Custom distribution plot
fig = plot_property_distribution(
    valid_df, 
    properties=['MW', 'LogP'],
    plot_type='histogram',  # or 'box'
    bins=20
)
fig.show()
```

### Comprehensive Dashboard

```python
from viz import create_molecular_dashboard

# Create comprehensive analysis dashboard
dashboard = create_molecular_dashboard(valid_df)
dashboard.show()

# Save dashboard
dashboard.write_html("molecular_dashboard.html")
```

## Network Visualizations

### Molecular Graph Networks

```python
from viz import plot_molecular_network

# Visualize molecular topology
molecules = [
    ("Ethanol", "CCO"),
    ("Benzene", "c1ccccc1"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
]

for name, smiles in molecules:
    mol = parse_smiles(smiles)
    fig = plot_molecular_network(mol)
    fig.update_layout(title=f"{name} - Molecular Network")
    fig.show()
    fig.write_html(f"{name.lower()}_network.html")
```

### Custom Network Styling

```python
# Advanced network visualization
mol = parse_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
fig = plot_molecular_network(
    mol,
    node_size=15,
    edge_width=3,
    layout='spring',  # or 'circular', 'random'
    show_atom_labels=True,
    color_by_element=True
)

fig.update_layout(
    title="Caffeine Molecular Network",
    showlegend=True,
    width=800,
    height=600
)

fig.show()
```

## Advanced Visualization Techniques

### Property Correlation Heatmaps

```python
import plotly.express as px
import pandas as pd

# Calculate correlation matrix
numeric_cols = ['MW', 'LogP', 'TPSA', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumRings']
corr_matrix = valid_df[numeric_cols].corr()

# Create heatmap
fig = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    title="Molecular Property Correlations"
)

fig.show()
```

### 3D Property Space

```python
import plotly.express as px

# 3D scatter plot in property space
fig = px.scatter_3d(
    valid_df,
    x='MW',
    y='LogP', 
    z='TPSA',
    hover_data=['SMILES'],
    title="Molecules in 3D Property Space",
    labels={
        'MW': 'Molecular Weight (Da)',
        'LogP': 'Lipophilicity',
        'TPSA': 'Polar Surface Area (Ų)'
    }
)

fig.show()
```

### Interactive Property Explorer

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_property_explorer(df):
    """Create interactive property exploration dashboard."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "MW vs LogP",
            "TPSA Distribution", 
            "Rotatable Bonds",
            "H-Bond Capacity"
        ],
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # MW vs LogP scatter
    fig.add_trace(
        go.Scatter(
            x=df['MW'],
            y=df['LogP'],
            mode='markers',
            text=df['SMILES'],
            name='Molecules',
            marker=dict(size=8, opacity=0.7)
        ),
        row=1, col=1
    )
    
    # TPSA histogram
    fig.add_trace(
        go.Histogram(
            x=df['TPSA'],
            name='TPSA',
            nbinsx=10
        ),
        row=1, col=2
    )
    
    # Rotatable bonds bar chart
    rot_bonds_counts = df['NumRotatableBonds'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=rot_bonds_counts.index,
            y=rot_bonds_counts.values,
            name='Rotatable Bonds'
        ),
        row=2, col=1
    )
    
    # H-bond donors vs acceptors
    fig.add_trace(
        go.Scatter(
            x=df['NumHBD'],
            y=df['NumHBA'],
            mode='markers',
            text=df['SMILES'],
            name='H-Bonds',
            marker=dict(size=8, opacity=0.7)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Interactive Molecular Property Explorer",
        height=800,
        showlegend=False
    )
    
    return fig

# Create and display explorer
explorer = create_property_explorer(valid_df)
explorer.show()
```

## Batch Visualization

### Multiple Molecule Grid

```python
from PIL import Image
import math

def create_molecule_grid(smiles_list, grid_cols=3, img_size=200):
    """Create grid of 2D molecular structures."""
    
    molecules = []
    for smiles in smiles_list:
        try:
            mol = parse_smiles(smiles)
            if mol is not None:
                img = draw_molecule_2d(mol, width=img_size, height=img_size)
                molecules.append((smiles, img))
        except:
            continue
    
    if not molecules:
        return None
    
    # Calculate grid dimensions
    grid_rows = math.ceil(len(molecules) / grid_cols)
    
    # Create combined image
    grid_width = grid_cols * img_size
    grid_height = grid_rows * img_size
    grid_img = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Paste individual images
    for i, (smiles, img) in enumerate(molecules):
        row = i // grid_cols
        col = i % grid_cols
        x = col * img_size
        y = row * img_size
        grid_img.paste(img, (x, y))
    
    return grid_img

# Create molecule grid
test_smiles = [
    "CCO", "CCC", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
]

grid = create_molecule_grid(test_smiles, grid_cols=3)
if grid:
    grid.save("molecule_grid.png")
    grid.show()
```

### Animated Property Evolution

```python
import plotly.express as px

def create_property_animation(df, x_prop='MW', y_prop='LogP', animation_frame='NumRings'):
    """Create animated scatter plot showing property relationships."""
    
    fig = px.scatter(
        df,
        x=x_prop,
        y=y_prop,
        animation_frame=animation_frame,
        hover_data=['SMILES'],
        title=f"{y_prop} vs {x_prop} by {animation_frame}",
        range_x=[df[x_prop].min()*0.9, df[x_prop].max()*1.1],
        range_y=[df[y_prop].min()*0.9, df[y_prop].max()*1.1]
    )
    
    return fig

# Create animated visualization
if len(valid_df) > 5:  # Only if we have enough data
    anim_fig = create_property_animation(valid_df)
    anim_fig.show()
```

## Export and Sharing

### High-Quality Static Images

```python
# Export high-resolution images
molecules = ["CCO", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]

for i, smiles in enumerate(molecules):
    mol = parse_smiles(smiles)
    
    # 2D high-res
    img_2d = draw_molecule_2d(mol, width=800, height=600)
    img_2d.save(f"molecule_{i}_2d_hires.png")
    
    # 3D static
    fig_3d = create_3d_conformer_plot(mol)
    fig_3d.write_image(f"molecule_{i}_3d_static.png", width=800, height=600)
```

### Interactive HTML Reports

```python
def create_molecular_report(smiles_list, output_file="molecular_report.html"):
    """Create comprehensive HTML report with all visualizations."""
    
    # Process molecules
    df = create_molecular_dataframe(smiles_list)
    valid_df = df[df['Valid']].copy()
    
    # Create visualizations
    dashboard = create_molecular_dashboard(valid_df)
    properties_plot = plot_molecular_properties(valid_df)
    
    # Combine into report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Molecular Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 30px 0; }}
            .plot {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Molecular Analysis Report</h1>
        
        <div class="section">
            <h2>Dataset Summary</h2>
            <p>Total molecules: {len(df)}</p>
            <p>Valid molecules: {len(valid_df)}</p>
            <p>Success rate: {len(valid_df)/len(df)*100:.1f}%</p>
        </div>
        
        <div class="section">
            <h2>Comprehensive Dashboard</h2>
            <div class="plot">{dashboard.to_html(include_plotlyjs='cdn')}</div>
        </div>
        
        <div class="section">
            <h2>Property Correlations</h2>
            <div class="plot">{properties_plot.to_html(include_plotlyjs='cdn')}</div>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Report saved to {output_file}")

# Generate report
report_smiles = [
    "CCO", "CCC", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
]

create_molecular_report(report_smiles)
```

## Best Practices

### Performance Optimization

1. **Use appropriate image sizes** - Don't create unnecessarily large images
2. **Batch processing** - Process multiple molecules together
3. **Cache results** - Save generated visualizations for reuse
4. **Memory management** - Clear large objects when done

### Visual Design

1. **Consistent styling** - Use consistent colors and sizes
2. **Clear labels** - Always label axes and provide titles
3. **Interactive elements** - Use hover information effectively
4. **Color accessibility** - Consider colorblind-friendly palettes

### File Management

```python
import os
from pathlib import Path

# Organize output files
output_dir = Path("molecular_visualizations")
output_dir.mkdir(exist_ok=True)

# Create subdirectories
(output_dir / "2d_structures").mkdir(exist_ok=True)
(output_dir / "3d_conformers").mkdir(exist_ok=True)
(output_dir / "property_plots").mkdir(exist_ok=True)
(output_dir / "networks").mkdir(exist_ok=True)

# Save with organized structure
molecules = ["CCO", "c1ccccc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]

for i, smiles in enumerate(molecules):
    mol = parse_smiles(smiles)
    name = f"molecule_{i}"
    
    # 2D structure
    img_2d = draw_molecule_2d(mol)
    img_2d.save(output_dir / "2d_structures" / f"{name}.png")
    
    # 3D conformer
    fig_3d = create_3d_conformer_plot(mol)
    fig_3d.write_html(output_dir / "3d_conformers" / f"{name}.html")
    
    # Network
    fig_net = plot_molecular_network(mol)
    fig_net.write_html(output_dir / "networks" / f"{name}_network.html")

print(f"All visualizations saved to {output_dir}")
```

This tutorial covers all visualization capabilities. Next, explore the [API Reference](api-reference.md) for detailed function documentation.
