"""Constants for vectorstore module"""

import re

INDEXES_TABLE_NAME = "vector_indexes"
INDEX_NAME_FIELD = "name"
INDEX_DIMENSION_FIELD = "dimension"
INDEX_DIMENSION_FIELD_DEFAULT = 1536
INDEX_METRIC_FIELD = "metric"
INDEX_DELETION_PROTECT_FIELD = "deletion_protection"
INDEX_USE_VECTOR_INDEX_FIELD = "use_vector_index"
INDEX_VECTOR_INDEX_OPTIONS_FIELD = "vector_index_options"
INDEX_TAGS_FIELD = "tags"


def _get_index_table_name(index: str) -> str:
    """Get the index table name."""
    # Replace any character that is not A-Z, a-z, 0-9, or _ with underscore
    sanitized_index = re.sub(r"[^A-Za-z0-9_]", "_", index)
    return f"index_{sanitized_index}"


ID_FIELD = "id"
VECTOR_FIELD = "vector"
METADATA_FIELD = "metadata"
VECTOR_NORMALIZED_FIELD = "vector_normalized"
NAMESPACE_FIELD = "namespace"

VECTOR_INDEX = "vector_index"

METRIC_TYPE = "metric_type"
