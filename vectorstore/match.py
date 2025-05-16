from typing import List, TypedDict


class MatchTypedDict(TypedDict):
    """
    A class representing a match with its associated metadata.
    """
    id: str
    score: float
    values: List[float]
    metadata: dict