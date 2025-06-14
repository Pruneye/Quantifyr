import pytest
from gnn_quantum.gnn_quantum import stub


def test_stub():
    assert callable(stub)
