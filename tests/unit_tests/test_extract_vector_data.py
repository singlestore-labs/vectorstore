"""
Unit tests for the _extract_vector_data method in the _Index class.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from vectorstore._index import _Index
from vectorstore.index_model import IndexModel
from vectorstore.vector import Vector


@pytest.fixture
def index_instance():
    """Create a test instance of _Index class."""
    index_model = IndexModel(name="test_index", dimension=128)
    return _Index(index=index_model, connection=MagicMock())


def test_extract_from_vector_class(index_instance):
    """Test extracting data from a Vector class instance."""
    vector_id = "test-id-1"
    vector_values = [1.0, 2.0, 3.0]
    vector_metadata = {"key1": "value1", "key2": 42}

    vector = Vector(id=vector_id, vector=vector_values, metadata=vector_metadata)

    extracted_id, extracted_vector, extracted_metadata = (
        index_instance._extract_vector_data(vector)
    )

    assert extracted_id == vector_id
    assert extracted_vector == "[1.0,2.0,3.0]"
    assert extracted_metadata == json.dumps(vector_metadata)


def test_extract_from_dictionary(index_instance):
    """Test extracting data from a dictionary format."""
    vector_dict = {
        "id": "test-id-2",
        "values": [4.0, 5.0, 6.0],
        "metadata": {"category": "test", "importance": 0.9},
    }

    extracted_id, extracted_vector, extracted_metadata = (
        index_instance._extract_vector_data(vector_dict)
    )

    assert extracted_id == vector_dict["id"]
    assert extracted_vector == "[4.0,5.0,6.0]"
    assert extracted_metadata == json.dumps(vector_dict["metadata"])


def test_extract_from_tuple_without_metadata(index_instance):
    """Test extracting data from a tuple with 2 elements (id, values)."""
    vector_tuple = ("test-id-3", [7.0, 8.0, 9.0])

    extracted_id, extracted_vector, extracted_metadata = (
        index_instance._extract_vector_data(vector_tuple)
    )

    assert extracted_id == vector_tuple[0]
    assert extracted_vector == "[7.0,8.0,9.0]"
    assert extracted_metadata is None


def test_extract_from_tuple_with_metadata(index_instance):
    """Test extracting data from a tuple with 3 elements (id, values, metadata)."""
    vector_tuple = ("test-id-4", [10.0, 11.0, 12.0], {"tags": ["important", "urgent"]})

    extracted_id, extracted_vector, extracted_metadata = (
        index_instance._extract_vector_data(vector_tuple)
    )

    assert extracted_id == vector_tuple[0]
    assert extracted_vector == "[10.0,11.0,12.0]"
    assert extracted_metadata == json.dumps(vector_tuple[2])


def test_invalid_tuple_length(index_instance):
    """Test error handling with invalid tuple length."""
    invalid_tuple = ("test-id", [1.0, 2.0], {"metadata": "value"}, "extra_element")

    with pytest.raises(ValueError, match="Invalid vector tuple length"):
        index_instance._extract_vector_data(invalid_tuple)


def test_unsupported_vector_type(index_instance):
    """Test error handling with unsupported vector type."""
    invalid_input = "not-a-vector"

    with pytest.raises(ValueError, match="Unsupported vector type"):
        index_instance._extract_vector_data(invalid_input)


def test_complex_metadata(index_instance):
    """Test extraction with complex nested metadata."""
    complex_metadata = {
        "string": "value",
        "number": 42,
        "bool": True,
        "null": None,
        "array": [1, 2, 3],
        "nested": {"a": "nested value", "b": [{"x": 1}, {"y": 2}]},
    }

    vector = Vector(id="complex", vector=[1.0], metadata=complex_metadata)

    _, _, extracted_metadata = index_instance._extract_vector_data(vector)

    # Verify that the metadata was properly JSON serialized and contains all expected elements
    parsed_metadata = json.loads(extracted_metadata)
    assert parsed_metadata == complex_metadata


def test_empty_vector_values(index_instance):
    """Test extraction with empty vector values."""
    vector = Vector(id="empty-vector", vector=[], metadata={"note": "empty vector"})

    extracted_id, extracted_vector, extracted_metadata = (
        index_instance._extract_vector_data(vector)
    )

    assert extracted_id == "empty-vector"
    assert extracted_vector == "[]"
    assert json.loads(extracted_metadata) == {"note": "empty vector"}


@patch("vectorstore._index._Index._format_vector_values")
def test_format_vector_values_called(mock_format, index_instance):
    """Test that _format_vector_values is called with correct arguments."""
    mock_format.return_value = "[formatted vector]"

    vector_values = [1.0, 2.0, 3.0]
    vector = Vector(id="test", vector=vector_values, metadata={})

    _, extracted_vector, _ = index_instance._extract_vector_data(vector)

    mock_format.assert_called_once_with(vector_values)
    assert extracted_vector == "[formatted vector]"


def test_different_vector_value_types(index_instance):
    """Test extraction with different vector value types (int, float, mixed)."""
    vectors = [
        Vector(id="int-vector", vector=[1, 2, 3], metadata={}),
        Vector(id="float-vector", vector=[1.5, 2.5, 3.5], metadata={}),
        Vector(id="mixed-vector", vector=[1, 2.5, 3], metadata={}),
    ]

    for vector in vectors:
        _, extracted_vector, _ = index_instance._extract_vector_data(vector)
        # Verify the formatted string contains all the original values
        for val in vector.vector:
            assert str(val) in extracted_vector
