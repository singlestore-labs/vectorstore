"""
Unit tests for the _format_vector_values method in the _Index class.
"""

from unittest.mock import MagicMock

import pytest

from vectorstore._index import _Index
from vectorstore.index_model import IndexModel


@pytest.fixture
def index_instance():
    """Create a test instance of _Index class."""
    index_model = IndexModel(name="test_index", dimension=128)
    return _Index(index=index_model, connection=MagicMock())


def test_format_vector_values_basic(index_instance):
    """Test basic formatting of vector values."""
    vector = [1, 2, 3, 4, 5]
    formatted = index_instance._format_vector_values(vector)
    assert formatted == "[1,2,3,4,5]"


def test_format_vector_values_float(index_instance):
    """Test formatting of float vector values."""
    vector = [1.5, 2.75, 3.333, 4.0, 5.1]
    formatted = index_instance._format_vector_values(vector)
    assert formatted == "[1.5,2.75,3.333,4.0,5.1]"


def test_format_vector_values_mixed(index_instance):
    """Test formatting of mixed integer and float values."""
    vector = [1, 2.5, 3, 4.75, 5]
    formatted = index_instance._format_vector_values(vector)
    assert formatted == "[1,2.5,3,4.75,5]"


def test_format_vector_values_empty(index_instance):
    """Test formatting of an empty vector."""
    vector = []
    formatted = index_instance._format_vector_values(vector)
    assert formatted == "[]"


def test_format_vector_values_single(index_instance):
    """Test formatting of a vector with a single value."""
    vector = [42]
    formatted = index_instance._format_vector_values(vector)
    assert formatted == "[42]"


def test_format_vector_values_scientific_notation(index_instance):
    """Test formatting of vector with very large/small values in scientific notation."""
    vector = [1e10, 1e-10, 1.23e5]
    formatted = index_instance._format_vector_values(vector)
    # Python's string representation of floats can vary, so we check that
    # the values are correctly interpreted when parsed back
    assert formatted.startswith("[") and formatted.endswith("]")
    numbers = formatted[1:-1].split(",")
    assert float(numbers[0]) == 1e10
    assert float(numbers[1]) == 1e-10
    assert float(numbers[2]) == 1.23e5


def test_format_vector_values_negative(index_instance):
    """Test formatting of vector with negative values."""
    vector = [-1, -2.5, -3.333]
    formatted = index_instance._format_vector_values(vector)
    assert formatted == "[-1,-2.5,-3.333]"


def test_format_vector_values_zeros(index_instance):
    """Test formatting of vector with zero values."""
    vector = [0, 0.0, 0]
    formatted = index_instance._format_vector_values(vector)
    assert formatted == "[0,0.0,0]"


def test_format_vector_values_special_cases(index_instance):
    """Test formatting of vectors with special values like infinity."""
    vector = [float("inf"), float("-inf")]
    formatted = index_instance._format_vector_values(vector)
    # The exact representation might vary by Python version
    assert "inf" in formatted.lower()
    assert "-inf" in formatted.lower()


def test_format_vector_values_long_vector(index_instance):
    """Test formatting of a long vector."""
    vector = list(range(100))  # 0 to 99
    formatted = index_instance._format_vector_values(vector)
    assert formatted.startswith("[0,1,2,3,")
    assert formatted.endswith(",97,98,99]")
    assert len(formatted.split(",")) == 100
