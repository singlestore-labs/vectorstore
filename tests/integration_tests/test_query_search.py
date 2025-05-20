"""
Tests for vector querying and similarity search operations.
"""

from typing import List

import pytest

from vectorstore.match import MatchTypedDict


def test_query_by_id(index_with_sample_data):
    """Test querying using an existing vector ID as reference."""
    results = index_with_sample_data.query(id="id3", top_k=1, include_values=True)
    assert len(results) == 1
    assert results[0]["id"] == "id3"
    assert results[0]["values"] == pytest.approx([0.7, 0.8, 0.9], rel=1e-3)


def test_query_validation_errors(index_cosine):
    """Test validation errors in query parameters."""
    # Test error when neither vector nor id is provided
    with pytest.raises(ValueError):
        index_cosine.query(
            top_k=10,
            namespace="test_namespace",
            include_metadata=True,
            include_values=True,
        )

    # Test error when both vector and id are provided
    with pytest.raises(ValueError):
        index_cosine.query(vector=[0.1, 0.2, 0.3], id="id0", top_k=2)


def test_query_dotproduct(index_dotproduct):
    """Test dot product similarity search."""
    index_dotproduct.upsert(
        [
            ("id1", [0.1, 0.2, 0.3]),
            ("id2", [1, 2, 3]),
            ("id3", [0.7, 0.8, 0.9]),
        ]
    )

    query_vector = [0.1, 0.2, 0.3]
    results: List[MatchTypedDict] = index_dotproduct.query(
        vector=query_vector, top_k=2, include_values=True
    )

    assert len(results) == 2
    assert results[0]["id"] == "id2"  # Highest dot product with query vector
    assert results[1]["id"] == "id3"
    assert results[0]["score"] == pytest.approx(1.4, rel=1e-3)
    assert results[1]["score"] == pytest.approx(0.5, rel=1e-3)


def test_query_cosine(index_cosine):
    """Test cosine similarity search."""
    index_cosine.upsert(
        [
            ("id1", [0.1, 0.2, 0.3]),
            ("id2", [1, 2, 3.3]),
            ("id3", [0.7, 0.8, 0.9]),
        ]
    )

    query_vector = [0.1, 0.2, 0.3]
    results: List[MatchTypedDict] = index_cosine.query(
        vector=query_vector, top_k=2, include_values=True
    )

    assert len(results) == 2
    assert results[0]["id"] == "id1"  # Most similar direction to query vector
    assert results[1]["id"] == "id2"
    assert results[0]["score"] == pytest.approx(1.0, rel=1e-3)
    assert results[1]["score"] == pytest.approx(0.998, rel=1e-3)


def test_query_euclidean(index_euclidean):
    """Test Euclidean distance similarity search."""
    index_euclidean.upsert(
        [
            ("id1", [0.1, 0.2, 0.3]),
            ("id2", [1, 2, 3]),
            ("id3", [0.7, 0.8, 0.9]),
        ]
    )

    query_vector = [0.1, 0.2, 0.3]
    results: List[MatchTypedDict] = index_euclidean.query(
        vector=query_vector, top_k=2, include_values=True
    )

    assert len(results) == 2
    assert results[0]["id"] == "id1"  # Smallest Euclidean distance to query vector
    assert results[1]["id"] == "id3"
    assert results[0]["score"] == pytest.approx(0.0, rel=1e-3)
    assert results[1]["score"] == pytest.approx(1.039, rel=1e-3)


def test_vector_index_search_cosine(cosine_index_with_vector_index):
    """Test search with cosine similarity and vector indexing."""
    # Insert multiple vectors
    cosine_index_with_vector_index.upsert(
        [
            (f"id{i}", [0.1 * i * 3, 0.1 * (i * 3 + 1), 0.1 * (i * 3 + 2)])
            for i in range(1, 1000)
        ]
    )

    query_vector = [0.1, 0.2, 0.3]
    results: List[MatchTypedDict] = cosine_index_with_vector_index.query(
        vector=query_vector, top_k=2, include_values=True
    )

    assert len(results) == 2
    assert results[0]["id"] == "id1"  # Most similar to query vector
    assert results[1]["id"] == "id2"
    assert results[0]["score"] == pytest.approx(0.982, rel=1e-3)
    assert results[1]["score"] == pytest.approx(0.963, rel=1e-3)


def test_vector_index_disable(dotproduct_index_with_vector_index):
    """Test disabling vector index for a specific query."""
    # Insert multiple vectors
    dotproduct_index_with_vector_index.upsert(
        [
            (f"id{i}", [0.1 * i * 3, 0.1 * (i * 3 + 1), 0.1 * (i * 3 + 2)])
            for i in range(1, 1000)
        ]
    )

    query_vector = [0.1, 0.2, 0.3]
    results: List[MatchTypedDict] = dotproduct_index_with_vector_index.query(
        vector=query_vector,
        top_k=2,
        include_values=False,
        disable_vector_index_use=True,
    )

    # When disabling the vector index, full scan is performed, which gives the exact top results
    assert len(results) == 2
    assert results[0]["id"] == "id999"  # Highest dot product with query vector
    assert results[1]["id"] == "id998"
    assert results[0]["score"] == pytest.approx(179.9, rel=1e-3)
    assert results[1]["score"] == pytest.approx(179.72, rel=1e-3)


def test_vector_index_options(euclidean_index_with_vector_index):
    """Test search with custom vector index options."""
    # Insert multiple vectors
    euclidean_index_with_vector_index.upsert(
        [
            (f"id{i}", [0.1 * i * 3, 0.1 * (i * 3 + 1), 0.1 * (i * 3 + 2)])
            for i in range(1, 1000)
        ]
    )

    query_vector = [0.1, 0.2, 0.3]
    results: List[MatchTypedDict] = euclidean_index_with_vector_index.query(
        vector=query_vector, top_k=2, include_values=False, search_options={"k": 10}
    )

    assert len(results) == 2
    assert results[0]["id"] == "id1"  # Closest by Euclidean distance
    assert results[1]["id"] == "id2"
    assert results[0]["score"] == pytest.approx(0.3464, rel=1e-3)
    assert results[1]["score"] == pytest.approx(0.866, rel=1e-3)


def test_vector_index_validation_errors(
    index_cosine, euclidean_index_with_vector_index
):
    """Test validation errors related to vector index usage."""
    # Test error when using search options with an index that doesn't have vector indexing
    with pytest.raises(ValueError):
        index_cosine.query(
            vector=[0.1, 0.2, 0.3],
            top_k=2,
            include_values=True,
            search_options={"k": 10},
        )

    # Test error when trying to use both disable_vector_index and search_options
    with pytest.raises(ValueError):
        euclidean_index_with_vector_index.query(
            vector=[0.1, 0.2, 0.3],
            top_k=2,
            disable_vector_index_use=True,
            search_options={"k": 10},
        )

    # Test error when trying to disable vector index on an index that doesn't have one
    with pytest.raises(ValueError):
        index_cosine.query(
            vector=[0.1, 0.2, 0.3], top_k=2, disable_vector_index_use=True
        )
