"""
Unit tests for the _get_where_clauses method in _Index class.
"""

from unittest.mock import MagicMock, patch

import pytest

from vectorstore._constants import ID_FIELD, NAMESPACE_FIELD
from vectorstore._index import _Index
from vectorstore.index_model import IndexModel


@pytest.fixture
def index_instance():
    """Create a test instance of _Index class."""
    index_model = IndexModel(name="test_index", dimension=128)
    return _Index(index=index_model, connection=MagicMock())


def test_empty_where_clauses(index_instance):
    """Test when no filtering parameters are provided."""
    fields, params = index_instance._get_where_clauses()
    assert len(fields) == 0
    assert len(params) == 0


def test_id_where_clause(index_instance):
    """Test filtering by a single ID."""
    fields, params = index_instance._get_where_clauses(id="test_id")

    assert len(fields) == 1
    assert fields[0] == f"{ID_FIELD} = %s"
    assert params == ["test_id"]


def test_prefix_where_clause(index_instance):
    """Test filtering by ID prefix."""
    fields, params = index_instance._get_where_clauses(prefix="test_prefix")

    assert len(fields) == 1
    assert fields[0] == f"{ID_FIELD} LIKE %s"
    assert params == ["test_prefix%"]


def test_namespace_where_clause(index_instance):
    """Test filtering by namespace."""
    fields, params = index_instance._get_where_clauses(namespace="test_namespace")

    assert len(fields) == 1
    assert fields[0] == f"{NAMESPACE_FIELD} = %s"
    assert params == ["test_namespace"]


def test_namespaces_where_clause(index_instance):
    """Test filtering by multiple namespaces."""
    test_namespaces = ["namespace1", "namespace2", "namespace3"]
    fields, params = index_instance._get_where_clauses(namespaces=test_namespaces)

    assert len(fields) == 1
    assert fields[0] == f"{NAMESPACE_FIELD} IN (%s,%s,%s)"
    assert params == test_namespaces


def test_empty_namespaces_where_clause(index_instance):
    """Test filtering with empty namespaces list."""
    fields, params = index_instance._get_where_clauses(namespaces=[])

    assert len(fields) == 0
    assert len(params) == 0


def test_ids_where_clause(index_instance):
    """Test filtering by multiple IDs."""
    test_ids = ["id1", "id2", "id3"]
    fields, params = index_instance._get_where_clauses(ids=test_ids)

    assert len(fields) == 1
    assert fields[0] == f"{ID_FIELD} IN (%s,%s,%s)"
    assert params == test_ids


@patch("vectorstore._index._parse_filter")
def test_filter_where_clause(mock_parse_filter, index_instance):
    """Test filtering with filter dictionary."""
    # Mock the _parse_filter function
    mock_filter_result = (
        "JSON_MATCH_ANY(metadata::?field, MATCH_PARAM_STRING_STRICT() = %s)",
        ["value"],
    )
    mock_parse_filter.return_value = mock_filter_result

    test_filter = {"field": "value"}
    fields, params = index_instance._get_where_clauses(filter=test_filter)

    mock_parse_filter.assert_called_once_with(test_filter)
    assert len(fields) == 1
    assert fields[0] == mock_filter_result[0]
    assert params == mock_filter_result[1]


def test_multiple_conditions_where_clause(index_instance):
    """Test combining multiple filtering conditions."""
    fields, params = index_instance._get_where_clauses(
        id="test_id", namespace="test_namespace"
    )

    assert len(fields) == 2
    assert f"{ID_FIELD} = %s" in fields
    assert f"{NAMESPACE_FIELD} = %s" in fields
    assert params == ["test_id", "test_namespace"]


@patch("vectorstore._index._parse_filter")
def test_complex_where_clause(mock_parse_filter, index_instance):
    """Test complex combination of filtering conditions."""
    mock_filter_result = (
        "JSON_MATCH_ANY(metadata::?field, MATCH_PARAM_STRING_STRICT() = %s)",
        ["value"],
    )
    mock_parse_filter.return_value = mock_filter_result

    test_ids = ["id1", "id2"]
    test_filter = {"field": "value"}

    fields, params = index_instance._get_where_clauses(
        ids=test_ids, namespace="test_namespace", filter=test_filter
    )

    assert len(fields) == 3
    assert f"{ID_FIELD} IN (%s,%s)" in fields
    assert f"{NAMESPACE_FIELD} = %s" in fields
    assert mock_filter_result[0] in fields

    # Check parameters - order matters based on how fields were added
    expected_params = ["test_namespace"] + test_ids + ["value"]
    assert params == expected_params


def test_namespace_conflict_validation(index_instance):
    """Test validation for namespace and namespaces conflict."""
    with pytest.raises(
        ValueError, match="Cannot specify both namespace and namespaces"
    ):
        index_instance._get_where_clauses(
            namespace="test", namespaces=["test1", "test2"]
        )


def test_namespaces_type_validation(index_instance):
    """Test validation for namespaces type."""
    with pytest.raises(ValueError, match="Namespaces must be a list"):
        index_instance._get_where_clauses(namespaces="not_a_list")


def test_all_parameters(index_instance):
    """Test with multiple parameters without conflict."""
    with patch("vectorstore._index._parse_filter") as mock_parse_filter:
        mock_filter_result = (
            "JSON_MATCH_ANY(metadata::?field, MATCH_PARAM_STRING_STRICT() = %s)",
            ["value"],
        )
        mock_parse_filter.return_value = mock_filter_result

        test_ids = ["id1", "id2"]
        test_filter = {"field": "value"}

        # Using multiple parameters without namespace/namespaces conflict
        fields, params = index_instance._get_where_clauses(
            id="test_id",
            prefix="test_prefix",
            namespace="test_namespace",  # Using only namespace, not namespaces
            ids=test_ids,
            filter=test_filter,
        )

        # Should have all conditions
        assert len(fields) == 5
        assert f"{ID_FIELD} = %s" in fields
        assert f"{ID_FIELD} LIKE %s" in fields
        assert f"{NAMESPACE_FIELD} = %s" in fields
        assert f"{ID_FIELD} IN (%s,%s)" in fields
        assert mock_filter_result[0] in fields

        # Parameters should be in the correct order based on how fields were added
        expected_params = (
            ["test_id", "test_prefix%", "test_namespace"] + test_ids + ["value"]
        )
        assert params == expected_params
