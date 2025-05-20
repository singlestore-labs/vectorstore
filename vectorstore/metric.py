from enum import Enum


class Metric(Enum):
    """
    The metric specifies how SingleStore should calculate the distance
    between vectors when querying an index
    Cosine - cosine distance calculated dot product of normalized vectors
    Euclidean - euclidean distance calculated as the square root of the sum of the squared differences
    DotProduct - dot product of the two vectors
    The default is cosine
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOTPRODUCT = "dotproduct"
