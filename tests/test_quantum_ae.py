import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quantum_ae.quantum_ae import stub


def test_stub():
    assert callable(stub)
