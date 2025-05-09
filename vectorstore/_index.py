import json
from typing import Any, Dict, List, Optional, Union, Tuple

from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool

from vectorstore.metric import Metric

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
                
                # Handle cosine similarity normalization
                if self.index.metric == Metric.COSINE:
                    fields.append(VECTOR_NORMALIZED_FIELD)
                
                # Process each vector
                table_name = _get_index_table_name(self.index.name)
                placeholders = ','.join(['%s'] * len(fields))
                
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
        self, df: Any, namespace: Optional[str] = None, batch_size: int = 500, show_progress: bool = True
    ):
        """Upserts a dataframe into the index.

        Args:
            df: A pandas dataframe with the following columns: id, values, sparse_values, and metadata.
            namespace: The namespace to upsert into.
            batch_size: The number of rows to upsert in a single batch.
            show_progress: Whether to show a progress bar.
        """
        # Implementation to be added
        pass

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

    def fetch(self, ids: List[str], namespace: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Implementation to be added
        return {}

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
        # Implementation to be added
        return {}

    def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> Any:
        # Implementation to be added
        return None

    def list(self, **kwargs) -> List[str]:
        # Implementation to be added
        return []
