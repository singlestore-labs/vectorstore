from typing import Dict, List, NamedTuple, TypedDict, Union


VectorDictMetadataValue = Union[str, int, float, List[str], List[int], List[float]]
VectorMetadataTypedDict = Dict[str, VectorDictMetadataValue]
VectorTuple = tuple[str, List[float]]
VectorTupleWithMetadata = tuple[str, List[float], VectorMetadataTypedDict]

class VectorTypedDict(TypedDict, total=False):
    values: List[float]
    metadata: dict
    id: str

class Vector(NamedTuple):
    id: str
    vector: List[float]
    metadata: VectorMetadataTypedDict

    def __repr__(self) -> str:
        return f"Vector(id={self.id}, vector={self.vector}, metadata={self.metadata})"