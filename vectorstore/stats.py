from typing import Dict, TypedDict

class NamespaceStatsTypedDict(TypedDict):
    vector_count: int

class IndexStatsTypedDict(TypedDict):
    dimension: int
    total_vector_count: int
    namespaces: Dict[str, NamespaceStatsTypedDict]