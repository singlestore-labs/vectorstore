from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from vectorstore.match import MatchTypedDict
from vectorstore.stats import IndexStatsTypedDict

from .filter import FilterTypedDict
from .vector import (
    Vector,
    VectorMetadataTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
    VectorTypedDict,
)


class IndexInterface(ABC):
    @abstractmethod
    def upsert(
        self,
        vectors: Union[
            List[Vector],
            List[VectorTuple],
            List[VectorTupleWithMetadata],
            List[VectorTypedDict],
        ],
        namespace: Optional[str] = None,
        **kwargs,
    ) -> int:
        """
        Insert or update vectors in the index.

        The upsert operation writes vectors into a namespace. If a vector already exists with the
        same ID, it will be overwritten. For SingleStore DB, this uses the REPLACE INTO SQL operation.

        Args:
            vectors: List of vectors in supported formats:
                - Vector objects
                - Tuples of (id, values) without metadata
                - Tuples of (id, values, metadata) with metadata
                - Dictionaries with 'id', 'values', and 'metadata' keys
            namespace: Optional namespace to organize vectors
            **kwargs: Additional arguments

        Returns:
            Number of vectors successfully upserted or updated

        """
        pass

    @abstractmethod
    def upsert_from_dataframe(
        self, df, namespace: Optional[str] = None, batch_size: int = 500
    ) -> int:
        """
        Upserts vectors from a pandas DataFrame into the index.

        The DataFrame is processed in batches to avoid memory issues with large datasets.
        Each row in the DataFrame should contain 'id', 'values', and optionally 'metadata'.

        Args:
            df: A pandas DataFrame with columns: 'id', 'values', and optionally 'metadata'
            namespace: Optional namespace to organize vectors
            batch_size: Number of rows to process in each batch (default: 500)

        Returns:
            Number of vectors successfully upserted
        """
        pass

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Delete vectors from the index based on provided criteria.

        Vectors can be deleted by ID, by filter, or all vectors in a namespace.
        The operation uses the SQL DELETE statement with appropriate WHERE clauses.

        Args:
            ids: List of vector IDs to delete
            delete_all: If True, delete all vectors (use with caution)
            namespace: Namespace to delete from
            filter: Filter condition for deletion using metadata
            **kwargs: Additional arguments

        Returns:
            Empty dict on success

        Raises:
            ValueError: If the deletion criteria are invalid or conflicting

        Note:
            - Cannot specify both ids and filter simultaneously
            - Cannot use delete_all with ids or filter
            - Must specify at least one of: ids, delete_all, or filter
        """
        pass

    @abstractmethod
    def fetch(
        self, ids: List[str], namespace: Optional[str] = None, **kwargs
    ) -> Dict[str, Vector]:
        """
        Fetch vectors by their IDs from the index.

        Retrieves complete vector data including IDs, vector values, and metadata
        for the specified vector IDs.

        Args:
            ids: List of vector IDs to retrieve
            namespace: Optional namespace to fetch from
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping vector IDs to Vector objects containing
            the complete vector data (id, values, and metadata)
        """
        pass

    @abstractmethod
    def query(
        self,
        *,
        top_k: int,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        disable_vector_index_use: Optional[bool] = None,
        search_options: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> List[MatchTypedDict]:
        """
        Query the index for vectors similar to the provided vector or ID.

        Performs similarity search using the configured distance metric (dot product,
        cosine similarity, or Euclidean distance). Results can be filtered by metadata.

        Args:
            top_k: Number of most similar vectors to return
            vector: Query vector values for similarity search
            id: ID of an existing vector to use as the query vector
            namespace: Single namespace to search in
            namespaces: Multiple namespaces to search in (mutually exclusive with namespace)
            filter: Metadata filter to apply during search
            include_values: Whether to include vector values in results
            include_metadata: Whether to include metadata in results
            disable_vector_index_use: If True, disable vector index use for this query
            search_options: Dictionary of search options for the vector index
            **kwargs: Additional arguments

        Returns:
            List of matching vectors with similarity scores and optional metadata/values

        Raises:
            ValueError: If neither vector nor id is provided, or if both are provided
            ValueError: If vector index configuration parameters are invalid
        """
        pass

    @abstractmethod
    def update(
        self,
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Update an existing vector's values and/or metadata.

        Updates values, metadata, or both for a vector with the specified ID.
        For SingleStoreDB, this uses the SQL UPDATE statement.

        Args:
            id: Vector ID to update
            values: New vector values to set (optional)
            set_metadata: New metadata to set (optional)
            namespace: Namespace containing the vector (optional)
            **kwargs: Additional arguments

        Returns:
            Empty dict on success

        Raises:
            ValueError: If neither values nor set_metadata is provided
        """
        pass

    @abstractmethod
    def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> IndexStatsTypedDict:
        """
        Get statistics about the index contents.

        Returns information about vector counts per namespace, dimension size,
        and total vector count. Optionally filter statistics by metadata.

        Args:
            filter: Metadata filter to limit statistics to matching vectors (optional)
            **kwargs: Additional arguments

        Returns:
            IndexStatsTypedDict containing:
              - dimension: Vector dimension size
              - total_vector_count: Total number of vectors across all namespaces
              - namespaces: Dictionary mapping namespace names to statistics
        """
        pass

    @abstractmethod
    def list(
        self, prefix: Optional[str] = None, namespace: Optional[str] = None
    ) -> List[str]:
        """
        List vector IDs in the index, optionally filtered by prefix and namespace.

        Retrieves all vector IDs that match the given prefix within the specified namespace.
        Results are sorted by ID.

        Args:
            prefix: Optional ID prefix to filter results
            namespace: Optional namespace to list from

        Returns:
            List of matching vector IDs sorted in ascending order
        """
        pass
