"""
Tests for vector management operations (list, fetch, update, delete).
"""

import pytest


def test_list(index):
    """Test listing vector IDs with various filters."""
    # Insert test data in two namespaces
    index.upsert(
        [
            ("id1", [0.1, 0.2, 0.3]),
            ("pid2", [0.4, 0.5, 0.6]),
            ("id3", [0.7, 0.8, 0.9]),
        ],
        namespace="namespace1",
    )

    index.upsert(
        [
            ("pid4", [0.1, 0.2, 0.3]),
            ("id5", [0.4, 0.5, 0.6]),
            ("pid6", [0.7, 0.8, 0.9]),
        ],
        namespace="namespace2",
    )

    # Test listing all vectors
    all_list = index.list()
    assert all_list == ["id1", "id3", "id5", "pid2", "pid4", "pid6"]

    # Test listing by namespace
    namespace1_list = index.list(namespace="namespace1")
    assert namespace1_list == ["id1", "id3", "pid2"]

    namespace2_list = index.list(namespace="namespace2")
    assert namespace2_list == ["id5", "pid4", "pid6"]

    # Test listing with non-existent namespace
    empty_list = index.list(namespace="namespace3")
    assert empty_list == []

    # Test listing with prefix
    empty_list_prefix = index.list(prefix="DD")
    assert empty_list_prefix == []

    prefix_list1 = index.list(prefix="id")
    assert prefix_list1 == ["id1", "id3", "id5"]

    prefix_list2 = index.list(prefix="pid")
    assert prefix_list2 == ["pid2", "pid4", "pid6"]

    # Test listing with both prefix and namespace
    prefix_namespace_list = index.list(prefix="id", namespace="namespace1")
    assert prefix_namespace_list == ["id1", "id3"]


def test_fetch_basic(index_with_sample_data):
    """Test basic fetch operation with specific IDs."""
    fetched_vectors = index_with_sample_data.fetch(["id1", "id2"])

    # Check the structure of the result
    assert set(fetched_vectors.keys()) == {"id1", "id2"}

    # Check id1 vector properties
    assert fetched_vectors["id1"].id == "id1"
    assert fetched_vectors["id1"].metadata == {"key1": "value1"}
    assert fetched_vectors["id1"].vector == pytest.approx([0.1, 0.2, 0.3], rel=1e-3)

    # Check id2 vector properties
    assert fetched_vectors["id2"].id == "id2"
    assert fetched_vectors["id2"].metadata == {"key2": "value2"}
    assert fetched_vectors["id2"].vector == pytest.approx([0.4, 0.5, 0.6], rel=1e-3)


def test_fetch_nonexistent(index_with_sample_data):
    """Test fetch with non-existent vector ID."""
    empty_fetch = index_with_sample_data.fetch(["nonexistent_id"])
    assert empty_fetch == {}


def test_fetch_namespace(index_with_sample_data):
    """Test fetch with namespace."""
    # Test with non-existent namespace
    empty_fetch_namespace = index_with_sample_data.fetch(
        ["id1"], namespace="nonexistent_namespace"
    )
    assert empty_fetch_namespace == {}

    # Test with correct namespace
    fetch_namespace = index_with_sample_data.fetch(["id1"], namespace="test_namespace")
    assert len(fetch_namespace) == 1
    assert "id1" in fetch_namespace
    assert fetch_namespace["id1"].id == "id1"
    assert fetch_namespace["id1"].metadata == {"key1": "value1"}
    assert fetch_namespace["id1"].vector == pytest.approx([0.1, 0.2, 0.3], rel=1e-3)


def test_update_values(index_with_sample_data):
    """Test updating vector values."""
    # Update existing vector
    result = index_with_sample_data.update("id1", values=[0.9, 0.8, 0.7])
    assert result == {}

    fetched_vectors = index_with_sample_data.fetch(["id1"])
    assert fetched_vectors["id1"].id == "id1"
    assert fetched_vectors["id1"].metadata == {"key1": "value1"}
    assert fetched_vectors["id1"].vector == pytest.approx([0.9, 0.8, 0.7], rel=1e-3)


def test_update_nonexistent(index_with_sample_data):
    """Test updating non-existent vector."""
    result = index_with_sample_data.update("nonexistent_id", values=[0.1, 0.2, 0.3])
    assert result == {}
    assert index_with_sample_data.fetch(["nonexistent_id"]) == {}


def test_update_metadata(index_with_sample_data):
    """Test updating vector metadata."""
    result = index_with_sample_data.update(
        "id1", values=[0.9, 0.8, 0.7], set_metadata={"key1": "new_value"}
    )
    assert result == {}

    fetched_vectors = index_with_sample_data.fetch(["id1"])
    assert fetched_vectors["id1"].id == "id1"
    assert fetched_vectors["id1"].metadata == {"key1": "new_value"}
    assert fetched_vectors["id1"].vector == pytest.approx([0.9, 0.8, 0.7], rel=1e-3)


def test_update_with_namespace(index_with_sample_data):
    """Test updating vector with namespace."""
    result = index_with_sample_data.update(
        "id1", set_metadata={}, namespace="test_namespace"
    )
    assert result == {}

    fetched_vectors = index_with_sample_data.fetch(["id1"], namespace="test_namespace")
    assert len(fetched_vectors) == 1
    assert fetched_vectors["id1"].id == "id1"
    assert fetched_vectors["id1"].metadata == {}
    assert fetched_vectors["id1"].vector == pytest.approx([0.1, 0.2, 0.3], rel=1e-3)


def test_update_validation(index_with_sample_data):
    """Test update validation errors."""
    # Error when trying to update vector without providing values or metadata
    with pytest.raises(ValueError):
        index_with_sample_data.update("id1")

    # Error when trying to update vector with empty ID
    with pytest.raises(ValueError):
        index_with_sample_data.update(None, values=[0.1, 0.2, 0.3])


def test_delete_by_id(index_with_sample_data):
    """Test deleting a vector by ID."""
    index_with_sample_data.delete(["id1"])
    fetched_vectors = index_with_sample_data.fetch(["id1"])
    assert fetched_vectors == {}
    assert index_with_sample_data.list() == ["id2", "id3"]


def test_delete_nonexistent(index_with_sample_data):
    """Test deleting a non-existent vector."""
    index_with_sample_data.delete(["nonexistent_id"])
    fetched_vectors = index_with_sample_data.fetch(["nonexistent_id"])
    assert fetched_vectors == {}
    assert index_with_sample_data.list() == ["id1", "id2", "id3"]


def test_delete_with_namespace(index_with_sample_data):
    """Test deleting a vector with namespace."""
    index_with_sample_data.delete(["id1"], namespace="test_namespace")
    fetched_vectors = index_with_sample_data.fetch(["id1"], namespace="test_namespace")
    assert fetched_vectors == {}
    assert index_with_sample_data.list(namespace="test_namespace") == ["id2", "id3"]


def test_delete_all_namespace(index_with_sample_data):
    """Test deleting all vectors in a namespace."""
    index_with_sample_data.delete(namespace="test_namespace", delete_all=True)
    fetched_vectors = index_with_sample_data.fetch(
        ["id1", "id2", "id3"], namespace="test_namespace"
    )
    assert fetched_vectors == {}
    assert index_with_sample_data.list(namespace="test_namespace") == []


def test_delete_all(index_with_sample_data):
    """Test deleting all vectors."""
    index_with_sample_data.delete([], delete_all=True)
    fetched_vectors = index_with_sample_data.fetch(["id1", "id2", "id3"])
    assert fetched_vectors == {}
    assert index_with_sample_data.list() == []


def test_delete_with_filter(index_with_sample_data):
    """Test deleting vectors with a metadata filter."""
    index_with_sample_data.delete(
        filter={"$or": [{"key1": "value1"}, {"key2": {"$exists": True}}]}
    )
    assert index_with_sample_data.list() == ["id3"]


def test_delete_validation(index_with_sample_data):
    """Test delete validation errors."""
    # Error when trying to delete without providing IDs or namespace
    with pytest.raises(ValueError):
        index_with_sample_data.delete()

    # Error when trying to delete with empty IDs
    with pytest.raises(ValueError):
        index_with_sample_data.delete([])

    # Error when trying to delete with empty namespace
    with pytest.raises(ValueError):
        index_with_sample_data.delete(namespace="mynamespace")

    # Error when trying to delete with empty IDs and namespace
    with pytest.raises(ValueError):
        index_with_sample_data.delete([], namespace="mynamespace")

    # Error when trying to delete with IDs and delete_all=True
    with pytest.raises(ValueError):
        index_with_sample_data.delete(["id1"], delete_all=True)

    # Error when trying to delete with filter and delete_all=True
    with pytest.raises(ValueError):
        index_with_sample_data.delete(
            namespace="test_namespace", delete_all=True, filter={"key": "value"}
        )
