"""
VectorStore: A vector database library for storing and querying vector embeddings.

This module provides classes and utilities for vector similarity search,
metadata filtering, and vector index management.
"""

from importlib import metadata

# Utilities
from .delete_protection import DeletionProtection

# Filter types
from .filter import (
    # Logical operators
    AndFilter,
    EqFilter,
    # Field filters
    ExactMatchFilter,
    # Base filter types
    FilterTypedDict,
    GteFilter,
    GtFilter,
    InFilter,
    LteFilter,
    LtFilter,
    NeFilter,
    NinFilter,
    OrFilter,
    SimpleFilter,
)
from .index_interface import IndexInterface
from .index_list import IndexList
from .index_model import IndexModel

# Result types
from .match import MatchTypedDict

# Similarity metrics and strategies
from .metric import Metric
from .stats import IndexStatsTypedDict, NamespaceStatsTypedDict

# Vector representations
from .vector import (
    Vector,
    VectorDictMetadataValue,
    VectorMetadataTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
)

# Core components
from .vectordb import VectorDB

# Version handling
try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata  # Avoid polluting the namespace

# Define public API
__all__ = [
    # Core components
    "VectorDB",
    "IndexInterface",
    "IndexModel",
    "IndexList",
    # Vector representations
    "Vector",
    "VectorTuple",
    "VectorTupleWithMetadata",
    "VectorMetadataTypedDict",
    "VectorDictMetadataValue",
    # Similarity metrics and strategies
    "Metric",
    # Filter types
    "FilterTypedDict",
    "SimpleFilter",
    "AndFilter",
    "OrFilter",
    "ExactMatchFilter",
    "EqFilter",
    "NeFilter",
    "GtFilter",
    "GteFilter",
    "LtFilter",
    "LteFilter",
    "InFilter",
    "NinFilter",
    # Result types
    "MatchTypedDict",
    "IndexStatsTypedDict",
    "NamespaceStatsTypedDict",
    # Utilities
    "DeletionProtection",
    # Version
    "__version__",
]
