import pytest
from data_utils.data_utils import stub


def test_stub():
    assert callable(stub)
