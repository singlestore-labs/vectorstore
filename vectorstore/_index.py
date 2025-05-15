import json
from typing import Any, Dict, List, Optional, Union, Tuple

from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool

from vectorstore.metric import Metric
from vectorstore.stats import IndexStatsTypedDict, NamespaceStatsTypedDict

from .index_model import IndexModel
from .filter import FilterTypedDict
from .index_interface import IndexInterface, VectorTypedDict
from .vector import (
    Vector,
    VectorMetadataTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
)

from ._constants import (
    ID_FIELD,
    VECTOR_FIELD,
    METADATA_FIELD,
    VECTOR_NORMALIZED_FIELD,
    NAMESPACE_FIELD,
    _get_index_table_name,
)

class _Index(IndexInterface):

    def _get_connection(self) -> Connection:
        """Get a connection to the database."""
        if self.connection:
            return self.connection
        elif self.connection_pool:
            return self.connection_pool.connect()
        else:
            raise ValueError("No connection or connection pool provided.")

    def _close_connection_if_needed(self, conn: Connection) -> None:
        """Close the connection if it is not part of a pool."""
        if not self.connection:
            conn.close()

    def __init__(self,
        *,
        index: IndexModel,
        connection: Optional[Connection] = None,
        connection_pool: Optional[Pool] = None):

        self.index = index
        self.connection = connection
        self.connection_pool = connection_pool

    def _format_vector_values(self, vector_values: List[float]) -> str:
        """Format vector values for SQL insertion."""
        return f"[{','.join(map(str, vector_values))}]"

    def _extract_vector_data(self, vector: Union[Vector, VectorTypedDict, VectorTuple, VectorTupleWithMetadata]) -> Tuple[str, str, Optional[str]]:
        """
        Extract id, vector values, and metadata from various vector formats.

        Returns:
            Tuple of (id, formatted_vector_values, metadata_json_or_None)
        """
        if isinstance(vector, Vector):
            return (
                vector.id,
                self._format_vector_values(vector.vector),
                json.dumps(vector.metadata)
            )
        elif isinstance(vector, dict):
            return (
                vector["id"],
                self._format_vector_values(vector["values"]),
                json.dumps(vector["metadata"])
            )
        elif isinstance(vector, tuple):
            if len(vector) == 2:
                return (
                    vector[0],
                    self._format_vector_values(vector[1]),
                    None
                )
            elif len(vector) == 3:
                return (
                    vector[0],
                    self._format_vector_values(vector[1]),
                    json.dumps(vector[2])
                )
            else:
                raise ValueError(f"Invalid vector tuple length: {len(vector)}. Expected 2 or 3 elements.")
        else:
            raise ValueError(f"Unsupported vector type: {type(vector)}. Expected Vector, dict, or tuple.")

    def _get_where_clauses(self,
                           *,
                           id: Optional[str] = None,
                           prefix: Optional[str] = None,
                           namespace: Optional[str] = None,
                           ids: Optional[List[str]] = None,
                           filter: Optional[FilterTypedDict] = None) -> tuple[List[str], List[Any]]:
        fields = []
        params = []
        if id:
            fields.append(f"{ID_FIELD} = %s")
            params.append(id)
        if prefix:
            fields.append(f"{ID_FIELD} LIKE %s")
            params.append(f"{prefix}%")
        if namespace:
            fields.append(f"{NAMESPACE_FIELD} = %s")
            params.append(namespace)
        if ids:
            fields.append(f"{ID_FIELD} IN ({','.join(['%s'] * len(ids))})")
            params.extend(ids)
        if filter:
            pass
        return fields, params

    def upsert(
        self,
        vectors: Union[
            List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]
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
        try:
            with conn.cursor() as curr:
                # Determine fields to include in the query
                fields = [ID_FIELD, VECTOR_FIELD]

                # Add optional fields
                if namespace:
                    fields.insert(0, NAMESPACE_FIELD)

                # Check if we need to include metadata
                sample_vector = vectors[0]
                has_metadata = (
                    isinstance(sample_vector, Vector) or
                    isinstance(sample_vector, dict) or
                    (isinstance(sample_vector, tuple) and len(sample_vector) == 3)
                )

                if has_metadata:
                    fields.append(METADATA_FIELD)

                placeholders = ','.join(['%s'] * len(fields))

                # Handle cosine similarity normalization
                if self.index.metric == Metric.COSINE:
                    fields.append(VECTOR_NORMALIZED_FIELD)
                    vector_from_json = 'JSON_ARRAY_PACK(%s)'
                    vector_length = f"sqrt(vector_elements_sum(vector_mul({vector_from_json}, {vector_from_json})))"
                    placeholders += f", scalar_vector_mul(1.0 / {vector_length}, {vector_from_json})"

                # Process each vector
                table_name = _get_index_table_name(self.index.name)

                for vector in vectors:
                    # Extract data from vector
                    vector_id, formatted_vector, metadata_json = self._extract_vector_data(vector)

                    # Build values list for the SQL query
                    values = []
                    if namespace:
                        values.append(namespace)
                    values.append(vector_id)
                    values.append(formatted_vector)
                    if has_metadata:
                        values.append(metadata_json)

                    if self.index.metric == Metric.COSINE:
                        values.append(formatted_vector)
                        values.append(formatted_vector)
                        values.append(formatted_vector)
                    # Execute the insert/replace
                    curr.execute(
                        f"REPLACE INTO {table_name} ({','.join(fields)}) VALUES ({placeholders})",
                        values
                    )

                # Optimize table if using vector index
                if self.index.use_vector_index:
                    curr.execute(f"OPTIMIZE TABLE {table_name} FLUSH")
        finally:
            self._close_connection_if_needed(conn)

        return len(vectors)

    def upsert_from_dataframe(
        self, df: Any, namespace: Optional[str] = None, batch_size: int = 500
    ) -> int:
        """Upserts a dataframe into the index.

        Args:
            df: A pandas dataframe with the following columns: id, values, and metadata.
            namespace: The namespace to upsert into.
            batch_size: The number of rows to upsert in a single batch.
        """
        result = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            vectors = []
            for _, row in batch.iterrows():
                vector = Vector(
                    id=row["id"],
                    vector=row["values"],
                    metadata=row.get("metadata", {}),
                )
                vectors.append(vector)
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
        # Implementation to be added
        return {}

    def fetch(self, ids: List[str], namespace: Optional[str] = None, **kwargs) -> Dict[str, Vector]:
        where_fields, params = self._get_where_clauses(namespace=namespace, ids=ids)
        where_clause = " AND ".join(where_fields)
        where_clause = f"WHERE {where_clause}"
        table_name = _get_index_table_name(self.index.name)
        sql = f"SELECT {ID_FIELD}, {VECTOR_FIELD}, {METADATA_FIELD} FROM {table_name} {where_clause}"
        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                curr.execute(sql, params)
                rows = curr.fetchall()
                vectors = {}
                for row in rows:
                    vector_id = row[0]
                    vector_values = row[1].tolist()
                    metadata = row[2]
                    vectors[vector_id] = Vector(id=vector_id, vector=vector_values, metadata=metadata)
                return vectors
        finally:
            self._close_connection_if_needed(conn)

    def query(
        self,
        *args,
        top_k: int,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        **kwargs,
    ) -> Any:
        # Implementation to be added
        return None

    def query_namespaces(
        self,
        vector: List[float],
        namespaces: List[str],
        top_k: Optional[int] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        **kwargs,
    ) -> Any:
        # Implementation to be added
        return None

    def update(
        self,
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        set_fields = []
        params = []
        if values is not None:
            set_fields.append(f"{VECTOR_FIELD} = %s")
            params.append(self._format_vector_values(values))
        if set_metadata is not None:
            set_fields.append(f"{METADATA_FIELD} = %s")
            params.append(json.dumps(set_metadata))
        if len(set_fields) == 0:
            raise ValueError("No valid fields provided for update.")
        where_fields, where_params = self._get_where_clauses(id=id, namespace=namespace)
        params.extend(where_params)
        if len(where_fields) == 0:
            raise ValueError("No valid fields provided for update.")
        sql = f"""
            UPDATE {_get_index_table_name(self.index.name)}
            SET {', '.join(set_fields)}
            WHERE {' AND '.join(where_fields)}
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                curr.execute(sql, params)
        finally:
            self._close_connection_if_needed(conn)
        return {}

    def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> IndexStatsTypedDict:
        where_fields, params = self._get_where_clauses(filter=filter)
        where_clause = " AND ".join(where_fields)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        else:
            where_clause = ""
        table_name = _get_index_table_name(self.index.name)
        sql = f"""
            SELECT {NAMESPACE_FIELD}, count(*) AS vector_count
            FROM {table_name} {where_clause} GROUP BY {NAMESPACE_FIELD};
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                curr.execute(sql, params)
                rows = curr.fetchall()
                namespaces = {}
                for row in rows:
                    namespace = row[0]
                    vector_count = row[1]
                    namespaces[namespace] = NamespaceStatsTypedDict(
                        vector_count=vector_count
                    )
                return IndexStatsTypedDict(
                    dimension=self.index.dimension,
                    total_vector_count=sum(ns["vector_count"] for ns in namespaces.values()),
                    namespaces=namespaces,
                )
        finally:
            self._close_connection_if_needed(conn)
        return None

    def list(self, prefix: Optional[str] = None, namespace: Optional[str] = None) -> List[str]:
        where_fields, params = self._get_where_clauses(prefix=prefix, namespace=namespace)
        where_clause = " AND ".join(where_fields)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        else:
            where_clause = ""
        table_name = _get_index_table_name(self.index.name)
        sql = f"SELECT {ID_FIELD} FROM {table_name} {where_clause} ORDER BY {ID_FIELD}"
        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                curr.execute(sql, params)
                rows = curr.fetchall()
                return [row[0] for row in rows]
        finally:
            self._close_connection_if_needed(conn)
        return []
