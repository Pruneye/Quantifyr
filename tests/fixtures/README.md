# Test Fixtures

This directory contains test data and mock objects for Quantifyr tests:

- `sample_molecules.sdf` - Sample molecular data files
- `test_configs.yaml` - Test configuration files
- `mock_data.py` - Python fixtures and mock objects

## Usage

```python
import pytest
from tests.fixtures.mock_data import sample_molecules

def test_molecular_processing():
    mol = sample_molecules['ethanol']
    # ... test implementation
```
