import pytest
import json
from vectorstore.filter import _get_match_param_function, _parse_filter
from vectorstore._constants import METADATA_FIELD

# Tests for _get_match_param_function
def test_get_match_param_function():
    # Test string values
    assert _get_match_param_function("test") == "MATCH_PARAM_STRING_STRICT()"

    # Test numeric values
    assert _get_match_param_function(10) == "MATCH_PARAM_DOUBLE_STRICT()"
    assert _get_match_param_function(10.5) == "MATCH_PARAM_DOUBLE_STRICT()"

    # Test boolean values
    assert _get_match_param_function(True) == "MATCH_PARAM_BOOL_STRICT()"
    assert _get_match_param_function(False) == "MATCH_PARAM_BOOL_STRICT()"

    # Test unsupported value
    with pytest.raises(ValueError, match="Unsupported value type"):
        _get_match_param_function([1, 2, 3])
    with pytest.raises(ValueError, match="Unsupported value type"):
        _get_match_param_function({"key": "value"})

# Tests for _parse_filter - Simple Filters
def test_exact_match_filter():
    filter_dict = {"field": "value"}
    query, params = _parse_filter(filter_dict)
    assert query == f"JSON_MATCH_ANY({METADATA_FIELD}::?field, MATCH_PARAM_STRING_STRICT() = %s)"
    assert params == ["value"]

    filter_dict = {"field": 123}
    query, params = _parse_filter(filter_dict)
    assert query == f"JSON_MATCH_ANY({METADATA_FIELD}::?field, MATCH_PARAM_DOUBLE_STRICT() = %s)"
    assert params == [123]

    filter_dict = {"field": True}
    query, params = _parse_filter(filter_dict)
    assert query == f"JSON_MATCH_ANY({METADATA_FIELD}::?field, MATCH_PARAM_BOOL_STRICT() = %s)"
    assert params == [True]

def test_eq_filter():
    filter_dict = {"field": {"$eq": "value"}}
    query, params = _parse_filter(filter_dict)
    assert query == f"JSON_MATCH_ANY({METADATA_FIELD}::?field, MATCH_PARAM_STRING_STRICT() = %s)"
    assert params == ["value"]

def test_ne_filter():
    filter_dict = {"field": {"$ne": "value"}}
    query, params = _parse_filter(filter_dict)
    assert query == f"NOT JSON_MATCH_ANY({METADATA_FIELD}::?field, MATCH_PARAM_STRING_STRICT() = %s) AND JSON_MATCH_ANY_EXISTS({METADATA_FIELD}, 'field')"
    assert params == ["value"]

# Tests for numeric comparison filters
def test_gt_filter():
    filter_dict = {"field": {"$gt": 10}}
    query, params = _parse_filter(filter_dict)
    assert query == f"JSON_EXTRACT_DOUBLE({METADATA_FIELD}, 'field') > %s"
    assert params == [10]

    with pytest.raises(ValueError, match=r"\$gt must be a numeric value"):
        _parse_filter({"field": {"$gt": "string"}})

def test_gte_filter():
    filter_dict = {"field": {"$gte": 10}}
    query, params = _parse_filter(filter_dict)
    assert query == f"JSON_EXTRACT_DOUBLE({METADATA_FIELD}, 'field') >= %s"
    assert params == [10]

    with pytest.raises(ValueError, match=r"\$gte must be a numeric value"):
        _parse_filter({"field": {"$gte": "string"}})

def test_lt_filter():
    filter_dict = {"field": {"$lt": 10}}
    query, params = _parse_filter(filter_dict)
    assert query == f"JSON_EXTRACT_DOUBLE({METADATA_FIELD}, 'field') < %s"
    assert params == [10]

    with pytest.raises(ValueError, match=r"\$lt must be a numeric value"):
        _parse_filter({"field": {"$lt": "string"}})

def test_lte_filter():
    filter_dict = {"field": {"$lte": 10}}
    query, params = _parse_filter(filter_dict)
    assert query == f"JSON_EXTRACT_DOUBLE({METADATA_FIELD}, 'field') <= %s"
    assert params == [10]

    with pytest.raises(ValueError, match=r"\$lte must be a numeric value"):
        _parse_filter({"field": {"$lte": "string"}})

# Tests for array filters
def test_in_filter():
    filter_dict = {"field": {"$in": ["value1", "value2"]}}
    query, params = _parse_filter(filter_dict)
    assert query == f"JSON_MATCH_ANY({METADATA_FIELD}::?field, JSON_ARRAY_CONTAINS_JSON(%s, MATCH_PARAM_JSON()))"
    assert params == [json.dumps(["value1", "value2"])]

    with pytest.raises(ValueError, match=r"\$in must be a list"):
        _parse_filter({"field": {"$in": "not_a_list"}})

def test_nin_filter():
    filter_dict = {"field": {"$nin": ["value1", "value2"]}}
    query, params = _parse_filter(filter_dict)
    assert query == f"NOT JSON_MATCH_ANY({METADATA_FIELD}::?field, JSON_ARRAY_CONTAINS_JSON(%s, MATCH_PARAM_JSON())) AND JSON_MATCH_ANY_EXISTS({METADATA_FIELD}, 'field')"
    assert params == [json.dumps(["value1", "value2"])]

    with pytest.raises(ValueError, match=r"\$nin must be a list"):
        _parse_filter({"field": {"$nin": "not_a_list"}})

# Tests for exists filter
def test_exists_filter():
    filter_dict = {"field": {"$exists": True}}
    query, params = _parse_filter(filter_dict)
    assert query == f"JSON_MATCH_ANY_EXISTS({METADATA_FIELD}, 'field')"
    assert params == []

    filter_dict = {"field": {"$exists": False}}
    query, params = _parse_filter(filter_dict)
    assert query == f"NOT JSON_MATCH_ANY_EXISTS({METADATA_FIELD}, 'field')"
    assert params == []

    with pytest.raises(ValueError, match=r"\$exists must be a boolean"):
        _parse_filter({"field": {"$exists": "not_a_bool"}})

# Tests for logical operators
def test_and_filter():
    filter_dict = {"$and": [{"field1": "value1"}, {"field2": "value2"}]}
    query, params = _parse_filter(filter_dict)
    expected_query = f"JSON_MATCH_ANY({METADATA_FIELD}::?field1, MATCH_PARAM_STRING_STRICT() = %s) AND JSON_MATCH_ANY({METADATA_FIELD}::?field2, MATCH_PARAM_STRING_STRICT() = %s)"
    assert query == expected_query
    assert params == ["value1", "value2"]

    with pytest.raises(ValueError, match=r"\$and must be a list of filters"):
        _parse_filter({"$and": "not_a_list"})

def test_or_filter():
    filter_dict = {"$or": [{"field1": "value1"}, {"field2": "value2"}]}
    query, params = _parse_filter(filter_dict)
    expected_query = f"(JSON_MATCH_ANY({METADATA_FIELD}::?field1, MATCH_PARAM_STRING_STRICT() = %s) OR JSON_MATCH_ANY({METADATA_FIELD}::?field2, MATCH_PARAM_STRING_STRICT() = %s))"
    assert query == expected_query
    assert params == ["value1", "value2"]

    with pytest.raises(ValueError, match=r"\$or must be a list of filters"):
        _parse_filter({"$or": "not_a_list"})

# Tests for nested filters
def test_nested_filters():
    filter_dict = {
        "$and": [
            {"field1": "value1"},
            {"$or": [{"field2": {"$gt": 10}}, {"field3": {"$exists": True}}]}
        ]
    }
    query, params = _parse_filter(filter_dict)
    expected_query = f"JSON_MATCH_ANY({METADATA_FIELD}::?field1, MATCH_PARAM_STRING_STRICT() = %s) AND (JSON_EXTRACT_DOUBLE({METADATA_FIELD}, 'field2') > %s OR JSON_MATCH_ANY_EXISTS({METADATA_FIELD}, 'field3'))"
    assert query == expected_query
    assert params == ["value1", 10]

# Tests for error cases
def test_filter_error_cases():
    # Test non-dictionary filter
    with pytest.raises(ValueError, match="Filter must be a dictionary"):
        _parse_filter([1, 2, 3])

    # Test empty filter
    with pytest.raises(ValueError, match="Filter must contain exactly one key"):
        _parse_filter({})

    # Test filter with multiple keys
    with pytest.raises(ValueError, match="Filter must contain exactly one key"):
        _parse_filter({"field1": "value1", "field2": "value2"})

    # Test field filter with multiple operators
    with pytest.raises(ValueError, match="Field filter must contain exactly one key"):
        _parse_filter({"field": {"$eq": "value1", "$ne": "value2"}})

    # Test unsupported operator
    with pytest.raises(ValueError, match="Unsupported operator"):
        _parse_filter({"field": {"$unsupported": "value"}})
