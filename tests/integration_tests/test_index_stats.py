"""
Tests for index statistics.
"""


def test_basic_stats(index_with_sample_data):
    """Test basic index statistics."""
    stats = index_with_sample_data.describe_index_stats()
    assert stats == {
        "dimension": 3,
        "total_vector_count": 3,
        "namespaces": {"test_namespace": {"vector_count": 3}},
    }


def test_stats_with_basic_filter(index_with_sample_data):
    """Test index statistics with a simple filter."""
    stats = index_with_sample_data.describe_index_stats(
        filter={"$or": [{"key1": "value1"}, {"key2": {"$exists": True}}]}
    )
    assert stats == {
        "dimension": 3,
        "total_vector_count": 2,
        "namespaces": {"test_namespace": {"vector_count": 2}},
    }


def test_stats_with_complex_filter(index):
    """Test index statistics with complex filters."""
    # Insert test data with varied metadata
    index.upsert(
        [
            (
                "id1",
                [0.1, 0.2, 0.3],
                {
                    "key1": "value1",
                    "key2": 192,
                    "key3": True,
                    "key4": ["item1", "item2"],
                },
            ),
            (
                "id2",
                [0.4, 0.5, 0.6],
                {
                    "key1": "value2",
                    "key2": 193,
                    "key3": False,
                    "key4": ["item3", "item4"],
                },
            ),
            (
                "id3",
                [0.7, 0.8, 0.9],
                {"key1": "value3", "key2": 200.58, "key4": ["item2", "item3"]},
            ),
        ],
        namespace="test_namespace",
    )

    # Test with AND filter
    stats1 = index.describe_index_stats(
        filter={
            "$and": [
                {"key1": {"$in": ["value1", "value2"]}},
                {"key4": {"$nin": ["item4", "item3"]}},
            ]
        }
    )
    assert stats1 == {
        "dimension": 3,
        "total_vector_count": 1,
        "namespaces": {"test_namespace": {"vector_count": 1}},
    }

    # Test with OR filter
    stats2 = index.describe_index_stats(
        filter={"$or": [{"key2": {"$eq": 200.58}}, {"key3": {"$exists": True}}]}
    )
    assert stats2 == {
        "dimension": 3,
        "total_vector_count": 3,
        "namespaces": {"test_namespace": {"vector_count": 3}},
    }
