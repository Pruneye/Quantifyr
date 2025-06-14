import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnn_quantum.gnn_quantum import stub

def test_stub():
    assert callable(stub) 