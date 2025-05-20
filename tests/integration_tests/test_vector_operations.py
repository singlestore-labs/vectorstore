"""
Tests for basic vector operations (upsert, insert from dataframe).
"""

import pandas as pd
import pytest
from singlestoredb import connect

from vectorstore import Metric, Vector


def test_upsert(clean_connection_params, index_metric):
    """Test upserting vectors in different formats."""
    vectors = [
        ("id1", [0.1, 0.2, 0.3]),
        ("id2", [0.4, 0.5, 0.6]),
        ("id3", [0.7, 0.8, 0.9]),
    ]
    index = index_metric[0]
    metric = index_metric[1]
    count = index.upsert(vectors)
    assert count == 3

    conn = connect(**clean_connection_params)
    curr = conn.cursor()
    curr.execute("select count(*) from index_test_index")
    result = curr.fetchone()
    assert result[0] == 3

    curr.execute("select * from index_test_index order by id")
    results = curr.fetchall()
    assert len(results) == 3
    for i, vector in enumerate(vectors):
        if metric == Metric.COSINE:
            assert len(results[i]) == 7
            length = sum(vector[1][j] ** 2 for j in range(3)) ** 0.5
            assert results[i][6] == pytest.approx(
                [vector[1][j] / length for j in range(3)], rel=1e-3
            )
        else:
            assert len(results[i]) == 6
        assert results[i][0] == vector[0]
        assert results[i][1] == pytest.approx(vector[1], rel=1e-3)
        assert results[i][2] is None
        assert results[i][3] is None
    curr.close()
    conn.close()


def test_upsert_with_metadata(clean_connection_params, index):
    """Test upserting vectors with metadata."""
    vectors = [
        ("id1", [0.1, 0.2, 0.3], {"key1": "value1"}),
        ("id2", [0.4, 0.5, 0.6], {"key2": "value2"}),
        ("id3", [0.7, 0.8, 0.9], {"key3": "value3"}),
    ]
    count = index.upsert(vectors)
    assert count == 3

    conn = connect(**clean_connection_params)
    curr = conn.cursor()
    curr.execute("select count(*) from index_test_index")
    result = curr.fetchone()
    assert result[0] == 3

    curr.execute("select * from index_test_index order by id")
    results = curr.fetchall()
    assert len(results) == 3
    for i, vector in enumerate(vectors):
        assert len(results[i]) >= 6
        assert results[i][0] == vector[0]
        assert results[i][1] == pytest.approx(vector[1], rel=1e-3)
        assert results[i][2] == vector[2]
        assert results[i][3] is None
    curr.close()
    conn.close()


def test_upsert_with_namespace(clean_connection_params, index):
    """Test upserting vectors with a namespace."""
    vectors = [
        ("id1", [0.1, 0.2, 0.3], {"key1": "value1"}),
        ("id2", [0.4, 0.5, 0.6], {"key2": "value2"}),
        ("id3", [0.7, 0.8, 0.9], {"key3": "value3"}),
    ]
    count = index.upsert(vectors, namespace="test_namespace")
    assert count == 3

    conn = connect(**clean_connection_params)
    curr = conn.cursor()
    curr.execute("select count(*) from index_test_index")
    result = curr.fetchone()
    assert result[0] == 3

    curr.execute("select * from index_test_index order by id")
    results = curr.fetchall()
    assert len(results) == 3
    for i, vector in enumerate(vectors):
        assert len(results[i]) >= 6
        assert results[i][0] == vector[0]
        assert results[i][1] == pytest.approx(vector[1], rel=1e-3)
        assert results[i][2] == vector[2]
        assert results[i][3] == "test_namespace"
    curr.close()
    conn.close()


def test_upset_vector_objects(clean_connection_params, index):
    """Test upserting Vector class objects."""
    vectors = [
        Vector(id="id1", vector=[0.1, 0.2, 0.3], metadata={"key1": "value1"}),
        Vector(id="id2", vector=[0.5, 0.6, 0.7], metadata={"key2": "value2"}),
        Vector(id="id3", vector=[0.8, 0.9, 0.10], metadata={"key3": "value3"}),
    ]
    count = index.upsert(vectors, namespace="test_namespace")
    assert count == 3

    conn = connect(**clean_connection_params)
    curr = conn.cursor()
    curr.execute("select count(*) from index_test_index")
    result = curr.fetchone()
    assert result[0] == 3

    curr.execute("select * from index_test_index order by id")
    results = curr.fetchall()
    assert len(results) == 3
    for i, vector in enumerate(vectors):
        assert len(results[i]) >= 6
        assert results[i][0] == vector.id
        assert results[i][1] == pytest.approx(vector.vector, rel=1e-3)
        assert results[i][2] == vector.metadata
        assert results[i][3] == "test_namespace"
    curr.close()
    conn.close()


def test_upsert_named_dicts(clean_connection_params, index):
    """Test upserting dictionaries with named fields."""
    vectors = [
        {"id": "id1", "values": [0.1, 0.2, 0.3], "metadata": {"key1": "value1"}},
        {"id": "id2", "values": [0.4, 0.5, 0.6], "metadata": {"key2": "value2"}},
        {"id": "id3", "values": [0.7, 0.8, 0.9], "metadata": {"key3": "value3"}},
    ]
    count = index.upsert(vectors)
    assert count == 3

    conn = connect(**clean_connection_params)
    curr = conn.cursor()
    curr.execute("select count(*) from index_test_index")
    result = curr.fetchone()
    assert result[0] == 3

    curr.execute("select * from index_test_index order by id")
    results = curr.fetchall()
    assert len(results) == 3
    for i, vector in enumerate(vectors):
        assert len(results[i]) >= 6
        assert results[i][0] == vector["id"]
        assert results[i][1] == pytest.approx(vector["values"], rel=1e-3)
        assert results[i][2] == vector["metadata"]
        assert results[i][3] is None
    curr.close()
    conn.close()


def test_upsert_from_df(clean_connection_params, index):
    """Test upserting vectors from a pandas DataFrame."""
    vectors = pd.DataFrame(
        {
            "id": ["id1", "id2", "id3", "id2"],
            "values": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.1, 0.11, 0.12],
            ],
            "metadata": [
                {"key1": "value1"},
                {"key2": "value2"},
                {"key3": "value3"},
                {"key4": "value4"},
            ],
        }
    )
    count = index.upsert_from_dataframe(vectors, namespace="test_namespace")
    assert count == 5

    conn = connect(**clean_connection_params)
    curr = conn.cursor()
    curr.execute("select count(*) from index_test_index")
    result = curr.fetchone()
    assert result[0] == 3  # Should be 3 because id2 is updated, not inserted twice

    curr.execute("select * from index_test_index order by id")
    results = curr.fetchall()
    assert len(results) == 3

    # Check that id2 has been updated with the new values
    id2_row = next(row for row in results if row[0] == "id2")
    assert id2_row[1] == pytest.approx([0.1, 0.11, 0.12], rel=1e-3)
    assert id2_row[2] == {"key4": "value4"}

    curr.close()
    conn.close()
