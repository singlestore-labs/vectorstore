"""
Internal implementation of vector index operations for SingleStore DB.

This module provides the _Index class that implements the IndexInterface
for vector similarity search operations using SingleStore's vector capabilities.
"""

# Standard library imports
import json
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool

# Local imports
from ._constants import (
    ID_FIELD,
    METADATA_FIELD,
    NAMESPACE_FIELD,
    VECTOR_FIELD,
    VECTOR_NORMALIZED_FIELD,
    _get_index_table_name,
)
from ._distance_strategy import DistanceStrategy
from .filter import FilterTypedDict, _parse_filter
from .index_interface import IndexInterface, VectorTypedDict
from .index_model import IndexModel
from .match import MatchTypedDict
from .metric import Metric
from .stats import IndexStatsTypedDict, NamespaceStatsTypedDict
from .vector import (
    Vector,
    VectorMetadataTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
)


class _Index(IndexInterface):
    """
    Internal implementation of vector index operations for SingleStore DB.

    This class implements the IndexInterface by translating high-level vector operations
    into SQL queries for SingleStore DB tables.
    """

    def __init__(
        self,
        *,
        index: IndexModel,
        connection: Optional[Connection] = None,
        connection_pool: Optional[Pool] = None,
    ) -> None:
        """
        Initialize the index implementation with database connection.

        Args:
            index: IndexModel containing index configuration
            connection: Direct database connection (optional)
            connection_pool: Connection pool for database access (optional)

        Note:
            At least one of connection or connection_pool must be provided
        """
        self.index = index
        self.connection = connection
        self.connection_pool = connection_pool

    def _get_connection(self) -> Connection:
        """
        Get a database connection from either direct connection or connection pool.

        Returns:
            Active database connection

        Raises:
            ValueError: If no connection or connection pool is available
        """
        if self.connection:
            return self.connection
        elif self.connection_pool:
            return self.connection_pool.connect()
        else:
            raise ValueError("No connection or connection pool provided.")

    def _close_connection_if_needed(self, conn: Connection) -> None:
        """
        Close the database connection if it was obtained from a pool.

        Args:
            conn: The database connection to potentially close
        """
        if not self.connection:
            conn.close()

    def _format_vector_values(self, vector_values: List[float]) -> str:
        """
        Format vector values as a JSON-like array string for SQL insertion.

        Args:
            vector_values: List of numeric vector components

        Returns:
            Formatted string representation of the vector (e.g. "[1.0,2.0,3.0]")
        """
        return f"[{','.join(map(str, vector_values))}]"

    def _extract_vector_data(
        self,
        vector: Union[Vector, VectorTypedDict, VectorTuple, VectorTupleWithMetadata],
    ) -> Tuple[str, str, Optional[str]]:
        """
        Extract ID, vector values, and metadata from various vector formats.

        Handles different input vector formats (Vector class, dictionaries, tuples)
        and extracts the common components needed for database operations.

        Args:
            vector: Vector in one of the supported formats

        Returns:
            Tuple containing:
              - ID string
              - Formatted vector values string
              - JSON-serialized metadata (or None if not present)

        Raises:
            ValueError: If vector format is invalid or unsupported
        """
        if isinstance(vector, Vector):
            # Handle Vector class instances
            return (
                vector.id,
                self._format_vector_values(vector.vector),
                json.dumps(vector.metadata),
            )
        elif isinstance(vector, dict):
            # Handle dictionary format
            return (
                vector["id"],
                self._format_vector_values(vector["values"]),
                json.dumps(vector["metadata"]),
            )
        elif isinstance(vector, tuple):
            # Handle tuple formats with different lengths
            if len(vector) == 2:
                # (id, values) format without metadata
                return (vector[0], self._format_vector_values(vector[1]), None)
            elif len(vector) == 3:
                # (id, values, metadata) format
                return (
                    vector[0],
                    self._format_vector_values(vector[1]),
                    json.dumps(vector[2]),
                )
            else:
                raise ValueError(
                    f"Invalid vector tuple length: {len(vector)}. Expected 2 or 3 elements."
                )
        else:
            # Unsupported format
            raise ValueError(
                f"Unsupported vector type: {type(vector)}. Expected Vector, dict, or tuple."
            )

    def _get_where_clauses(
        self,
        *,
        id: Optional[str] = None,
        prefix: Optional[str] = None,
        namespace: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        filter: Optional[FilterTypedDict] = None,
    ) -> Tuple[List[str], List[Any]]:
        """
        Build WHERE clause components and parameters for SQL queries.

        This helper method generates SQL WHERE clause fragments and their
        corresponding parameter values based on the filtering criteria.

        Args:
            id: Single ID to filter by
            prefix: ID prefix to match
            namespace: Namespace to filter by
            namespaces: List of namespaces to include
            ids: List of IDs to include
            filter: Metadata filter specification

        Returns:
            Tuple containing:
              - List of WHERE clause fragments
              - List of parameter values to be bound to the query

        Raises:
            ValueError: If incompatible filter options are specified
        """
        fields = []
        params = []

        # Add ID exact match condition
        if id:
            fields.append(f"{ID_FIELD} = %s")
            params.append(id)

        # Add ID prefix match condition
        if prefix:
            fields.append(f"{ID_FIELD} LIKE %s")
            params.append(f"{prefix}%")

        # Add namespace filter
        if namespace:
            fields.append(f"{NAMESPACE_FIELD} = %s")
            params.append(namespace)

        # Add multi-namespace filter
        if namespaces is not None and len(namespaces) > 0:
            # Validate namespaces parameter
            if not isinstance(namespaces, list):
                raise ValueError("Namespaces must be a list")

            # Check for conflict with single namespace parameter
            if namespace:
                raise ValueError("Cannot specify both namespace and namespaces")

            # Create IN clause with the right number of placeholders
            fields.append(
                f"{NAMESPACE_FIELD} IN ({','.join(['%s'] * len(namespaces))})"
            )
            params.extend(namespaces)

        # Add multi-ID filter
        if ids:
            fields.append(f"{ID_FIELD} IN ({','.join(['%s'] * len(ids))})")
            params.extend(ids)

        # Add metadata filter
        if filter:
            # Convert filter dictionary to SQL WHERE clause and parameters
            filter_str, filter_params = _parse_filter(filter)
            fields.append(filter_str)
            params.extend(filter_params)

        return fields, params

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

        Args:
            vectors: List of vectors in various formats
            namespace: Optional namespace to organize vectors
            **kwargs: Additional arguments

        Returns:
            Number of vectors upserted
        """
        if not vectors:
            return 0

        conn = self._get_connection()
        result = 0
        try:
            with conn.cursor() as curr:
                # Determine fields to include in the query
                fields = [ID_FIELD, VECTOR_FIELD]

                # Add optional fields
                if namespace:
                    fields.insert(0, NAMESPACE_FIELD)

                # Check if we need to include metadata by examining the first vector
                sample_vector = vectors[0]
                has_metadata = (
                    isinstance(sample_vector, Vector)
                    or isinstance(sample_vector, dict)
                    or (isinstance(sample_vector, tuple) and len(sample_vector) == 3)
                )

                if has_metadata:
                    fields.append(METADATA_FIELD)

                # Create SQL placeholder string
                placeholders = ",".join(["%s"] * len(fields))

                # Handle cosine similarity normalization
                if self.index.metric == Metric.COSINE:
                    fields.append(VECTOR_NORMALIZED_FIELD)
                    vector_from_json = "JSON_ARRAY_PACK(%s)"
                    # Calculate vector length for normalization
                    vector_length = f"sqrt(vector_elements_sum(vector_mul({vector_from_json}, {vector_from_json})))"
                    # Add placeholder for normalized vector
                    placeholders += f", scalar_vector_mul(1.0 / {vector_length}, {vector_from_json})"

                # Get table name
                table_name = _get_index_table_name(self.index.name)

                # Process each vector
                for vector in vectors:
                    # Extract data from vector
                    vector_id, formatted_vector, metadata_json = (
                        self._extract_vector_data(vector)
                    )

                    # Build values list for the SQL query
                    values = []
                    if namespace:
                        values.append(namespace)
                    values.append(vector_id)
                    values.append(formatted_vector)
                    if has_metadata:
                        values.append(metadata_json)

                    # For cosine similarity, add vector values for normalization
                    if self.index.metric == Metric.COSINE:
                        # Add the same formatted vector multiple times (for each reference in the SQL)
                        values.append(formatted_vector)
                        values.append(formatted_vector)
                        values.append(formatted_vector)

                    # Execute the insert/replace
                    curr.execute(
                        f"REPLACE INTO {table_name} ({','.join(fields)}) VALUES ({placeholders})",
                        values,
                    )
                    result += curr.rowcount

                # Optimize table if using vector index
                if self.index.use_vector_index:
                    curr.execute(f"OPTIMIZE TABLE {table_name} FLUSH")
        finally:
            self._close_connection_if_needed(conn)

        return result

    def upsert_from_dataframe(
        self, df: Any, namespace: Optional[str] = None, batch_size: int = 500
    ) -> int:
        """
        Upsert vectors from a pandas DataFrame into the index.

        Args:
            df: A pandas DataFrame with columns: id, values, and optionally metadata
            namespace: Optional namespace to organize vectors
            batch_size: Number of rows to process in each batch

        Returns:
            Number of vectors upserted
        """
        result = 0
        # Process DataFrame in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            vectors = []

            # Convert DataFrame rows to Vector objects
            for _, row in batch.iterrows():
                vector = Vector(
                    id=row["id"],
                    vector=row["values"],
                    metadata=row.get("metadata", {}),
                )
                vectors.append(vector)

            # Upsert the batch
            result += self.upsert(vectors, namespace=namespace)

        return result

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

        Args:
            ids: List of vector IDs to delete
            delete_all: If True, delete all vectors (use with caution)
            namespace: Namespace to delete from
            filter: Filter condition for deletion
            **kwargs: Additional arguments

        Returns:
            Empty dict on success

        Raises:
            ValueError: If the deletion criteria are invalid
        """
        # Validate inputs
        has_ids = ids is not None and len(ids) > 0
        has_filter = filter is not None and len(filter) > 0

        # Check for conflicting parameters
        if delete_all is True and (has_ids or has_filter):
            raise ValueError(
                "Cannot delete all vectors and specify ids or filter at the same time"
            )

        if has_ids and has_filter:
            raise ValueError("Cannot specify both ids and filter for deletion")

        if not delete_all and not has_ids and not has_filter:
            raise ValueError(
                "Must specify ids, delete_all, or filter to delete vectors"
            )

        # Build WHERE clause
        where_fields, params = self._get_where_clauses(
            ids=ids, namespace=namespace, filter=filter
        )

        # Create full WHERE clause if needed
        where_clause = ""
        if where_fields:
            where_clause = "WHERE " + " AND ".join(where_fields)

        # Execute the delete
        table_name = _get_index_table_name(self.index.name)
        sql = f"DELETE FROM {table_name} {where_clause}"

        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                curr.execute(sql, params)
                return {}
        finally:
            self._close_connection_if_needed(conn)

    def fetch(
        self, ids: List[str], namespace: Optional[str] = None, **kwargs
    ) -> Dict[str, Vector]:
        """
        Fetch vectors by their IDs from the index.

        Args:
            ids: List of vector IDs to retrieve
            namespace: Optional namespace to fetch from
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping vector IDs to Vector objects
        """
        # Build WHERE clause
        where_fields, params = self._get_where_clauses(namespace=namespace, ids=ids)
        where_clause = " AND ".join(where_fields)
        where_clause = f"WHERE {where_clause}"

        # Build query
        table_name = _get_index_table_name(self.index.name)
        sql = f"SELECT {ID_FIELD}, {VECTOR_FIELD}, {METADATA_FIELD} FROM {table_name} {where_clause}"

        # Execute query
        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                curr.execute(sql, params)
                rows = curr.fetchall()

                # Build result dictionary
                vectors = {}
                for row in rows:
                    vector_id = row[0]
                    vector_values = row[1].tolist()
                    metadata = row[2]
                    vectors[vector_id] = Vector(
                        id=vector_id, vector=vector_values, metadata=metadata
                    )

                return vectors
        finally:
            self._close_connection_if_needed(conn)

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

        Args:
            top_k: Number of most similar vectors to return
            vector: Query vector values
            id: ID of an existing vector to use as query vector
            namespace: Single namespace to search in
            namespaces: Multiple namespaces to search in
            filter: Metadata filter to apply
            include_values: Whether to include vector values in results
            include_metadata: Whether to include metadata in results
            disable_vector_index_use: If True, disable vector index
            search_options: Options for vector index search
            **kwargs: Additional arguments

        Returns:
            List of matching vectors with similarity scores

        Raises:
            ValueError: If query parameters are invalid
        """
        # Validate query parameters
        if not vector and not id:
            raise ValueError("Must provide either a vector or an id for querying")
        if vector and id:
            raise ValueError("Cannot provide both a vector and an id for querying")

        # Validate vector index parameters
        disable_vector_index = (
            disable_vector_index_use and disable_vector_index_use is True
        )
        if self.index.use_vector_index is False and disable_vector_index:
            raise ValueError(
                "Vector index is not enabled for this index, cannot disable it"
            )
        if search_options and disable_vector_index:
            raise ValueError(
                "Cannot provide search options when vector index is disabled"
            )
        if search_options and not self.index.use_vector_index:
            raise ValueError(
                "Search options can only be used with vector index enabled"
            )

        # Build WHERE clause
        where_fields, params = self._get_where_clauses(
            namespace=namespace, namespaces=namespaces, filter=filter
        )

        # If querying by ID, get the vector values
        if id:
            vectors = self.fetch([id])
            if len(vectors) == 0:
                raise ValueError(f"Vector with id {id} not found.")
            vector = vectors[id].vector

        # Determine distance function based on metric
        distance_function = "-" + DistanceStrategy.DOT_PRODUCT.value
        if self.index.metric == Metric.EUCLIDEAN:
            distance_function = DistanceStrategy.EUCLIDEAN_DISTANCE.value

        # Determine which vector field to use
        value_field = VECTOR_FIELD
        if self.index.metric == Metric.COSINE:
            value_field = VECTOR_NORMALIZED_FIELD
            # Normalize the query vector for cosine similarity
            vector_length = sum(x**2 for x in vector) ** 0.5
            if vector_length > 0:
                vector = [x / vector_length for x in vector]

        # Format the vector for SQL
        formatted_vector = self._format_vector_values(vector)

        # Build WHERE clause string
        where_clause = " AND ".join(where_fields)
        where_clause = f"WHERE {where_clause}" if where_clause else ""

        # Get table name
        table_name = _get_index_table_name(self.index.name)

        # Determine vector index usage options
        vector_index_options = ""
        if disable_vector_index:
            vector_index_options = "USE INDEX ()"
        elif search_options:
            vector_index_options = f"SEARCH_OPTIONS '{json.dumps(search_options)}'"

        # Build the query
        sql = f"""
            SELECT {ID_FIELD}, {VECTOR_FIELD}, {METADATA_FIELD},
                   {distance_function}({value_field}, %s) AS __score
            FROM {table_name} {where_clause}
            ORDER BY __score {vector_index_options}
            LIMIT %s
        """

        # Add parameters
        params.insert(0, formatted_vector)
        params.append(top_k)

        # Execute query
        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                curr.execute(sql, params)
                rows = curr.fetchall()

                # Build results
                matches = []
                for row in rows:
                    match: MatchTypedDict = {
                        "id": row[0],
                        "values": row[1].tolist() if include_values else None,
                        "metadata": row[2] if include_metadata else None,
                        "score": -row[3]
                        if self.index.metric != Metric.EUCLIDEAN
                        else row[3],
                    }
                    matches.append(match)
                return matches
        finally:
            self._close_connection_if_needed(conn)

        return []  # Ensure we always return a list

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

        Args:
            id: Vector ID to update
            values: New vector values
            set_metadata: New metadata
            namespace: Namespace containing the vector
            **kwargs: Additional arguments

        Returns:
            Empty dict on success

        Raises:
            ValueError: If no fields provided for update
        """
        # Build SET clause
        set_fields = []
        params = []

        # Add vector values if provided
        if values is not None:
            set_fields.append(f"{VECTOR_FIELD} = %s")
            params.append(self._format_vector_values(values))

        # Add metadata if provided
        if set_metadata is not None:
            set_fields.append(f"{METADATA_FIELD} = %s")
            params.append(json.dumps(set_metadata))

        # Validate we have fields to update
        if not set_fields:
            raise ValueError("No valid fields provided for update.")

        # Build WHERE clause
        where_fields, where_params = self._get_where_clauses(id=id, namespace=namespace)
        params.extend(where_params)

        # Validate WHERE clause
        if not where_fields:
            raise ValueError("No valid fields provided for update.")

        # Build UPDATE statement
        table_name = _get_index_table_name(self.index.name)
        sql = f"""
            UPDATE {table_name}
            SET {", ".join(set_fields)}
            WHERE {" AND ".join(where_fields)}
        """

        # Execute update
        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                curr.execute(sql, params)
                return {}
        finally:
            self._close_connection_if_needed(conn)

    def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> IndexStatsTypedDict:
        """
        Get statistics about the index contents.

        Args:
            filter: Metadata filter to limit statistics
            **kwargs: Additional arguments

        Returns:
            IndexStatsTypedDict with statistics about the index
        """
        # Build WHERE clause
        where_fields, params = self._get_where_clauses(filter=filter)
        where_clause = " AND ".join(where_fields)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        else:
            where_clause = ""

        # Build query
        table_name = _get_index_table_name(self.index.name)
        sql = f"""
            SELECT {NAMESPACE_FIELD}, count(*) AS vector_count
            FROM {table_name} {where_clause} GROUP BY {NAMESPACE_FIELD};
        """

        # Execute query
        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                curr.execute(sql, params)
                rows = curr.fetchall()

                # Build namespace stats
                namespaces = {}
                for row in rows:
                    namespace = row[0]
                    vector_count = row[1]
                    namespaces[namespace] = NamespaceStatsTypedDict(
                        vector_count=vector_count
                    )

                # Create and return index stats
                return IndexStatsTypedDict(
                    dimension=self.index.dimension,
                    total_vector_count=sum(
                        ns["vector_count"] for ns in namespaces.values()
                    ),
                    namespaces=namespaces,
                )
        finally:
            self._close_connection_if_needed(conn)

        # Return empty stats if query fails
        return IndexStatsTypedDict(
            dimension=self.index.dimension,
            total_vector_count=0,
            namespaces={},
        )

    def list(
        self, prefix: Optional[str] = None, namespace: Optional[str] = None
    ) -> List[str]:
        """
        List vector IDs in the index, optionally filtered by prefix and namespace.

        Args:
            prefix: Optional ID prefix to filter results
            namespace: Optional namespace to list from

        Returns:
            List of matching vector IDs sorted in ascending order
        """
        # Build WHERE clause
        where_fields, params = self._get_where_clauses(
            prefix=prefix, namespace=namespace
        )
        where_clause = " AND ".join(where_fields)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        else:
            where_clause = ""

        # Build query
        table_name = _get_index_table_name(self.index.name)
        sql = f"SELECT {ID_FIELD} FROM {table_name} {where_clause} ORDER BY {ID_FIELD}"

        # Execute query
        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                curr.execute(sql, params)
                rows = curr.fetchall()
                return [row[0] for row in rows]
        finally:
            self._close_connection_if_needed(conn)

        return []
