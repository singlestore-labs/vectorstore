"""
Filter module for VectorStore.

This module provides typed filter definitions and utilities to convert
filter dictionaries into SQL query fragments for metadata filtering.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Tuple, Union

from vectorstore._constants import METADATA_FIELD

# Type definitions for filter values
FieldValue = Union[str, int, float, bool]
NumericFieldValue = Union[int, float]

# Simple filter types
ExactMatchFilter = Dict[str, FieldValue]

# Comparison operators
EqFilter = Dict[Literal["$eq"], FieldValue]
NeFilter = Dict[Literal["$ne"], FieldValue]
GtFilter = Dict[Literal["$gt"], NumericFieldValue]
GteFilter = Dict[Literal["$gte"], NumericFieldValue]
LtFilter = Dict[Literal["$lt"], NumericFieldValue]
LteFilter = Dict[Literal["$lte"], NumericFieldValue]

# Collection operators
InFilter = Dict[Literal["$in"], List[FieldValue]]
NinFilter = Dict[Literal["$nin"], List[FieldValue]]

# Existence operator
ExistsFilter = Dict[Literal["$exists"], bool]

# Combined filter types
FieldFilter = Union[
    EqFilter,
    NeFilter,
    GtFilter,
    GteFilter,
    LtFilter,
    LteFilter,
    InFilter,
    NinFilter,
    ExistsFilter,
]

SimpleFilter = Union[
    ExactMatchFilter,
    Dict[str, FieldFilter],
]

# Logical operators
AndFilter = Dict[Literal["$and"], List["FilterTypedDict"]] 
OrFilter = Dict[Literal["$or"], List["FilterTypedDict"]]

# Overall filter type
FilterTypedDict = Union[SimpleFilter, AndFilter, OrFilter]


def _get_match_param_function(value: Any) -> str:
    """
    Determine the appropriate match parameter function based on value type.

    Args:
        value: The value to match against

    Returns:
        String representing the SQL match function to use

    Raises:
        ValueError: If the value type is not supported
    """
    # Check bool first since bool is a subclass of int
    if isinstance(value, bool):
        return "MATCH_PARAM_BOOL_STRICT()"
    elif isinstance(value, str):
        return "MATCH_PARAM_STRING_STRICT()"
    elif isinstance(value, (int, float)):
        return "MATCH_PARAM_DOUBLE_STRICT()"
    else:
        raise ValueError(f"Unsupported value type: {type(value)}")


def _parse_filter(filter_dict: FilterTypedDict) -> Tuple[str, List[Any]]:
    """
    Parse a filter dictionary into an SQL query fragment and parameters.

    Args:
        filter_dict: Filter specification following the FilterTypedDict schema

    Returns:
        Tuple containing:
          - SQL query fragment string
          - List of parameter values to be substituted into the query

    Raises:
        ValueError: If the filter format is invalid
    """
    if not isinstance(filter_dict, dict):
        raise ValueError("Filter must be a dictionary")

    if len(filter_dict) != 1:
        raise ValueError("Filter must contain exactly one key")

    # Handle logical operators
    if "$and" in filter_dict:
        return _handle_and_filter(filter_dict)
    elif "$or" in filter_dict:
        return _handle_or_filter(filter_dict)
    else:
        # Handle field filters
        field_name = next(iter(filter_dict))
        field_value = filter_dict[field_name]

        if isinstance(field_value, dict):
            # Handle operator-based field filter
            return _handle_operator_filter(field_name, field_value)
        else:
            # Handle exact match filter
            match_func = _get_match_param_function(field_value)
            return (
                f"JSON_MATCH_ANY({METADATA_FIELD}::?{field_name}, {match_func} = %s)",
                [field_value]
            )


def _handle_and_filter(filter_dict: AndFilter) -> Tuple[str, List[Any]]:
    """Handle $and operator filter."""
    sub_filters = filter_dict["$and"]
    if not isinstance(sub_filters, list):
        raise ValueError("$and must be a list of filters")

    # Process each sub-filter
    parsed_filters = [_parse_filter(sub_filter) for sub_filter in sub_filters]

    # Join conditions with AND
    query = " AND ".join(item[0] for item in parsed_filters)

    # Flatten parameter lists
    params = [param for parsed_filter in parsed_filters for param in parsed_filter[1]]

    return query, params


def _handle_or_filter(filter_dict: OrFilter) -> Tuple[str, List[Any]]:
    """Handle $or operator filter."""
    sub_filters = filter_dict["$or"]
    if not isinstance(sub_filters, list):
        raise ValueError("$or must be a list of filters")

    # Process each sub-filter
    parsed_filters = [_parse_filter(sub_filter) for sub_filter in sub_filters]

    # Join conditions with OR and wrap in parentheses
    query = "(" + " OR ".join(item[0] for item in parsed_filters) + ")"

    # Flatten parameter lists
    params = [param for parsed_filter in parsed_filters for param in parsed_filter[1]]

    return query, params


def _handle_operator_filter(field_name: str, field_filter: Dict) -> Tuple[str, List[Any]]:
    """Handle operator-based field filters like $eq, $gt, etc."""
    if len(field_filter) != 1:
        raise ValueError("Field filter must contain exactly one key")

    operator = next(iter(field_filter))
    field_value = field_filter[operator]

    # Equal operator
    if operator == "$eq":
        match_func = _get_match_param_function(field_value)
        return (
            f"JSON_MATCH_ANY({METADATA_FIELD}::?{field_name}, {match_func} = %s)",
            [field_value]
        )

    # Not equal operator
    elif operator == "$ne":
        match_func = _get_match_param_function(field_value)
        return (
            f"NOT JSON_MATCH_ANY({METADATA_FIELD}::?{field_name}, {match_func} = %s) AND "
            f"JSON_MATCH_ANY_EXISTS({METADATA_FIELD}, '{field_name}')",
            [field_value]
        )

    # Numeric comparison operators
    elif operator in ("$gt", "$gte", "$lt", "$lte"):
        if not isinstance(field_value, (int, float)):
            raise ValueError(f"{operator} must be a numeric value")

        comparison_op = {
            "$gt": ">",
            "$gte": ">=",
            "$lt": "<",
            "$lte": "<="
        }[operator]

        return (
            f"JSON_EXTRACT_DOUBLE({METADATA_FIELD}, '{field_name}') {comparison_op} %s",
            [field_value]
        )

    # Collection operators
    elif operator in ("$in", "$nin"):
        if not isinstance(field_value, list):
            raise ValueError(f"{operator} must be a list")

        if operator == "$in":
            return (
                f"JSON_MATCH_ANY({METADATA_FIELD}::?{field_name}, "
                f"JSON_ARRAY_CONTAINS_JSON(%s, MATCH_PARAM_JSON()))",
                [json.dumps(field_value)]
            )
        else:  # $nin
            return (
                f"NOT JSON_MATCH_ANY({METADATA_FIELD}::?{field_name}, "
                f"JSON_ARRAY_CONTAINS_JSON(%s, MATCH_PARAM_JSON())) AND "
                f"JSON_MATCH_ANY_EXISTS({METADATA_FIELD}, '{field_name}')",
                [json.dumps(field_value)]
            )

    # Existence operator
    elif operator == "$exists":
        if not isinstance(field_value, bool):
            raise ValueError("$exists must be a boolean")

        if field_value:
            return f"JSON_MATCH_ANY_EXISTS({METADATA_FIELD}, '{field_name}')", []
        else:
            return f"NOT JSON_MATCH_ANY_EXISTS({METADATA_FIELD}, '{field_name}')", []

    else:
        raise ValueError(f"Unsupported operator: {operator}")