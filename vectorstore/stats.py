from typing import Dict, TypedDict


class NamespaceStatsTypedDict(TypedDict):
    """Namespace statistics"""

    vector_count: int


class IndexStatsTypedDict(TypedDict):
    """Index statistics"""

    dimension: int
    total_vector_count: int
    namespaces: Dict[str, NamespaceStatsTypedDict]
