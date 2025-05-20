from enum import Enum


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors.
    DOT_PRODUCT is used for dotpoduct and cosine (when applied to normalized vectors) distance.
    Euclidean distance is used for euclidean distance.
    """

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    DOT_PRODUCT = "DOT_PRODUCT"
