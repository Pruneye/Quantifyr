import pytest
from viz.viz import stub


def test_stub():
    assert callable(stub)
