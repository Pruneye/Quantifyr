import pytest
from quantum_ae.quantum_ae import stub


def test_stub():
    assert callable(stub)
