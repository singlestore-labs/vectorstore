"""
Tests for metadata filtering in queries.
"""

import math

import pytest


def test_filter_exact_match(index_cosine):
    """Test filtering with implicit equality."""
    index_cosine.upsert(
        [
            ("id1", [0.1, 0.2, 0.3], {"key": "A"}),
            ("id2", [0.4, 0.5, 0.6], {"key": "B"}),
            ("id3", [0.7, 0.8, 0.9], {"key": "C"}),
            ("id4", [0.1, 0.2, 0.3], {"key": ["A", "C"]}),
            ("id5", [0.4, 0.5, 0.6], {"key": True}),
            ("id6", [0.7, 0.8, 0.9], {"key": False}),
            ("id7", [0.1, 0.2, 0.3], {}),
            ("id8", [0.4, 0.5, 0.6], {"key": None}),
            ("id9", [0.7, 0.8, 0.9], {"key": [True, False]}),
            ("id10", [0.1, 0.2, 0.3], {"key": 5.1}),
            ("id11", [0.4, 0.5, 0.6], {"key": 5.5}),
            ("id12", [0.7, 0.8, 0.9], {"key": [6, 7, 5.1]}),
            ("id13", [0.1, 0.2, 0.3], {"key": 5}),
            ("id14", [0.4, 0.5, 0.6], {"key": 7}),
            ("id15", [0.7, 0.8, 0.9], {"key": [5, 6]}),
        ]
    )
    query_vector = [0.1, 0.2, 0.3]

    # Test string equality
    results = index_cosine.query(vector=query_vector, top_k=20, filter={"key": "C"})
    assert len(results) == 2
    assert results[0]["id"] == "id4"
    assert results[1]["id"] == "id3"

    # Test boolean equality
    results = index_cosine.query(vector=query_vector, top_k=20, filter={"key": True})
    assert len(results) == 2
    assert results[0]["id"] == "id5"
    assert results[1]["id"] == "id9"

    # Test float equality
    results = index_cosine.query(vector=query_vector, top_k=20, filter={"key": 5.1})
    assert len(results) == 2
    assert results[0]["id"] == "id10"
    assert results[1]["id"] == "id12"

    # Test integer equality
    results = index_cosine.query(vector=query_vector, top_k=20, filter={"key": 7})
    assert len(results) == 2
    assert results[0]["id"] == "id14"
    assert results[1]["id"] == "id12"


def test_filter_explicit_equality(index_cosine):
    """Test filtering with explicit equality operator."""
    index_cosine.upsert(
        [
            ("id1", [0.1, 0.2, 0.3], {"key": "A"}),
            ("id2", [0.4, 0.5, 0.6], {"key": "B"}),
            ("id3", [0.7, 0.8, 0.9], {"key": "C"}),
            ("id4", [0.1, 0.2, 0.3], {"key": ["A", "C"]}),
        ]
    )
    query_vector = [0.1, 0.2, 0.3]

    results = index_cosine.query(
        vector=query_vector, top_k=20, filter={"key": {"$eq": "C"}}
    )
    assert len(results) == 2
    assert results[0]["id"] == "id4"
    assert results[1]["id"] == "id3"


def test_filter_not_equal(index_cosine):
    """Test filtering with not equal operator."""
    index_cosine.upsert(
        [
            ("id1", [0.1, 0.2, 0.3], {"key": "A"}),
            ("id2", [0.4, 0.5, 0.6], {"key": "B"}),
            ("id3", [0.7, 0.8, 0.9], {"key": "C"}),
        ]
    )
    query_vector = [0.1, 0.2, 0.3]

    results = index_cosine.query(
        vector=query_vector, top_k=20, filter={"key": {"$ne": "A"}}
    )
    assert len(results) == 2
    assert results[0]["id"] == "id2"
    assert results[1]["id"] == "id3"


def test_filter_in_operator(index_cosine):
    """Test filtering with $in operator."""
    index_cosine.upsert(
        [
            ("id1", [0.1, 0.2, 0.3], {"key": "A"}),
            ("id2", [0.4, 0.5, 0.6], {"key": "B"}),
            ("id3", [0.7, 0.8, 0.9], {"key": "C"}),
            ("id4", [1, 1.1, 1.2], {"key": ["B", "C"]}),
            ("id5", [1.3, 1.4, 1.5], {"key": True}),
        ]
    )
    query_vector = [0.1, 0.2, 0.3]

    results = index_cosine.query(
        vector=query_vector, top_k=20, filter={"key": {"$in": ["A", "C"]}}
    )
    assert len(results) == 3
    assert results[0]["id"] == "id1"
    assert results[1]["id"] == "id3"
    assert results[2]["id"] == "id4"


def test_filter_not_in(index_cosine):
    """Test filtering with $nin operator."""
    index_cosine.upsert(
        [
            ("id1", [0.1, 0.2, 0.3], {"key": "A"}),
            ("id2", [0.4, 0.5, 0.6], {"key": "B"}),
            ("id3", [0.7, 0.8, 0.9], {"key": "C"}),
            ("id4", [1, 1.1, 1.2], {"key": ["B", "C"]}),
            ("id5", [1.9, 2, 2.1], {}),
        ]
    )
    query_vector = [0.1, 0.2, 0.3]

    results = index_cosine.query(
        vector=query_vector, top_k=20, filter={"key": {"$nin": ["A", "C"]}}
    )
    assert len(results) == 1
    assert results[0]["id"] == "id2"


def test_filter_exists(index_cosine):
    """Test filtering with $exists operator."""
    index_cosine.upsert(
        [
            ("id1", [0.1, 0.2, 0.3], {"key": "A"}),
            ("id2", [0.4, 0.5, 0.6], {"key": "B"}),
            ("id3", [0.7, 0.8, 0.9], {"key": "C"}),
            ("id4", [1, 1.1, 1.2], {"key": ["B", "C"]}),
            ("id5", [1.9, 2, 2.1], {}),
        ]
    )
    query_vector = [0.1, 0.2, 0.3]

    # Test exists=True
    results = index_cosine.query(
        vector=query_vector, top_k=20, filter={"key": {"$exists": True}}
    )
    assert len(results) == 4
    assert {r["id"] for r in results} == {"id1", "id2", "id3", "id4"}

    # Test exists=False
    results = index_cosine.query(
        vector=query_vector, top_k=20, filter={"key": {"$exists": False}}
    )
    assert len(results) == 1
    assert results[0]["id"] == "id5"


def test_filter_numeric_comparisons(index_cosine):
    """Test filtering with numeric comparison operators."""
    index_cosine.upsert(
        [
            ("id1", [0.1, 0.2, 0.3], {"key": 198}),
            ("id2", [0.4, 0.5, 0.6], {"key": 203}),
            ("id3", [0.7, 0.8, 0.9], {"key": 188}),
            ("id4", [1, 1.1, 1.2], {"key": 200}),
            ("id5", [1.9, 2, 2.1], {}),
        ]
    )
    query_vector = [0.1, 0.2, 0.3]

    # Test less than
    results = index_cosine.query(
        vector=query_vector, top_k=20, filter={"key": {"$lt": 200}}
    )
    assert len(results) == 2
    assert {r["id"] for r in results} == {"id1", "id3"}

    # Test less than or equal
    results = index_cosine.query(
        vector=query_vector, top_k=20, filter={"key": {"$lte": 200}}
    )
    assert len(results) == 3
    assert {r["id"] for r in results} == {"id1", "id3", "id4"}

    # Test greater than
    results = index_cosine.query(
        vector=query_vector, top_k=20, filter={"key": {"$gt": 200}}
    )
    assert len(results) == 1
    assert results[0]["id"] == "id2"

    # Test greater than or equal
    results = index_cosine.query(
        vector=query_vector, top_k=20, filter={"key": {"$gte": 200}}
    )
    assert len(results) == 2
    assert {r["id"] for r in results} == {"id2", "id4"}


def test_filter_logical_operators(index_cosine):
    """Test filtering with logical operators $and and $or."""
    index_cosine.upsert(
        [
            ("id1", [0.1, 0.2, 0.3], {"key": 198}),
            ("id2", [0.4, 0.5, 0.6], {"key": 203}),
            ("id3", [0.7, 0.8, 0.9], {"key": 188}),
            ("id4", [1, 1.1, 1.2], {"key": 200}),
            ("id5", [1.9, 2, 2.1], {}),
        ]
    )
    query_vector = [0.1, 0.2, 0.3]

    # Test $and operator
    results = index_cosine.query(
        vector=query_vector,
        top_k=20,
        filter={"$and": [{"key": {"$lt": 200}}, {"key": {"$gte": 198}}]},
    )
    assert len(results) == 1
    assert results[0]["id"] == "id1"

    # Test $or operator
    results = index_cosine.query(
        vector=query_vector,
        top_k=20,
        filter={"$or": [{"key": {"$lt": 198}}, {"key": {"$gt": 200}}]},
    )
    assert len(results) == 2
    assert {r["id"] for r in results} == {"id2", "id3"}

    # Test nested logical operators
    results = index_cosine.query(
        vector=query_vector,
        top_k=20,
        filter={
            "$or": [
                {"$and": [{"key": {"$lte": 200}}, {"key": {"$gte": 200}}]},
                {"key": {"$exists": False}},
            ]
        },
    )
    assert len(results) == 2
    assert {r["id"] for r in results} == {"id4", "id5"}


def test_complex_query_filter(index_euclidean):
    """Test complex query with multiple filter conditions."""
    index_euclidean.upsert(
        [
            (
                "id1",
                [-1, -1, -1],
                {
                    "a": [1, 2, 3],
                    "b": 6,
                    "c": 7.9,
                    "d": True,
                    "e": "action",
                    "f": 1,
                    "h": [9, 10],
                },
            ),
            (
                "id2",
                [-1, -1, 1],
                {"a": 3, "b": 5, "c": -5, "d": "drama", "e": "comedy"},
            ),
            (
                "id3",
                [-1, 1, -1],
                {"a": [2, 3], "b": 5.5, "c": 6.7, "d": ["drama", True], "e": False},
            ),
            ("id4", [-1, 1, 1], {"a": 3, "b": 4, "c": 8.1, "d": "action"}),
            ("id5", [1, -1, -1], {"a": 4, "b": 3, "d": "comedy", "g": "extra"}),
            ("id6", [1, -1, 1], {"b": 6, "c": 7.9, "d": True}),
            (
                "id7",
                [1, 1, -1],
                {"a": [1, 2], "c": 6.7, "d": ["action", "comedy"], "e": 5},
            ),
            ("id8", [1, 1, 1], {"a": 7, "b": 8, "c": 9.1, "d": "action", "e": False}),
        ]
    )

    query_vector = [0, 0, 0]
    results = index_euclidean.query(
        vector=query_vector,
        top_k=5,
        include_values=False,
        filter={
            "$and": [
                {"a": 3},
                {"b": {"$gt": 4.5}},
                {"c": {"$lte": 7.9}},
                {"d": {"$in": ["drama", True]}},
                {"e": {"$nin": [False, 5, "game"]}},
                {"f": {"$exists": True}},
                {"g": {"$exists": False}},
                {"h": {"$ne": 8}},
            ]
        },
    )

    # This filter should only match id1
    assert len(results) == 1
    assert results[0]["id"] == "id1"
    assert results[0]["score"] == pytest.approx(math.sqrt(3.0), rel=1e-3)
