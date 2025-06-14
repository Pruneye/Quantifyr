import pytest
from qvae.qvae import stub


def test_stub():
    assert callable(stub)
