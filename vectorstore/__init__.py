from importlib import metadata

from .delete_protection import DeletionProtection
from .distance_strategy import DistanceStrategy
from .index_list import IndexList
from .index_model import IndexModel
from .metric import Metric
from .index_interface import IndexInterface
from .vectordb import VectorDB
from .vector import Vector, VectorTuple, VectorTupleWithMetadata, VectorMetadataTypedDict, VectorDictMetadataValue
from .filter import SimpleFilter, AndFilter, FilterTypedDict, ExactMatchFilter, EqFilter, NeFilter, GtFilter, GteFilter, LtFilter, LteFilter, InFilter, NinFilter

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "VectorDB",
    "IndexModel",
    "IndexList",
    "IndexInterface",
    "DeletionProtection",
    "Metric",
    "DistanceStrategy",
    "Vector",
    "VectorTuple",
    "VectorTupleWithMetadata",
    "VectorMetadataTypedDict",
    "VectorDictMetadataValue",
    "SimpleFilter",
    "AndFilter",
    "FilterTypedDict",
    "ExactMatchFilter",
    "EqFilter",
    "NeFilter",
    "GtFilter",
    "GteFilter",
    "LtFilter",
    "LteFilter",
    "InFilter",
    "NinFilter",
    "__version__",
]
