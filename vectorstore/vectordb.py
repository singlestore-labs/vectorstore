"""
VectorDB: Main entry point for vector database operations.

This module provides the core functionality for managing vector indexes,
including creation, configuration, and access to the underlying data structures.
"""

# Standard library imports
from importlib import metadata
import json
from abc import ABC
from typing import Dict, Optional, Union

# Third-party imports
from singlestoredb import connect
from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool, QueuePool

# Local imports
from ._constants import (
    ID_FIELD,
    INDEX_DELETION_PROTECT_FIELD,
    INDEX_DIMENSION_FIELD,
    INDEX_DIMENSION_FIELD_DEFAULT,
    INDEX_METRIC_FIELD,
    INDEX_NAME_FIELD,
    INDEX_TAGS_FIELD,
    INDEX_USE_VECTOR_INDEX_FIELD,
    INDEX_VECTOR_INDEX_OPTIONS_FIELD,
    INDEXES_TABLE_NAME,
    METADATA_FIELD,
    METRIC_TYPE,
    NAMESPACE_FIELD,
    VECTOR_FIELD,
    VECTOR_INDEX,
    VECTOR_NORMALIZED_FIELD,
    _get_index_table_name,
)
from ._distance_strategy import DistanceStrategy
from ._index import _Index
from .delete_protection import DeletionProtection
from .index_interface import IndexInterface
from .index_list import IndexList
from .index_model import IndexModel
from .metric import Metric


class VectorDB(ABC):
    """
    Main class for interacting with vector database functionality.

    Provides methods for managing vector indexes and accessing vector data.
    Supports connection pooling for efficient database access.
    """

    def __init__(
        self,
        *,
        connection: Optional[Connection] = None,
        connection_pool: Optional[Pool] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        **kwargs,
    ):
        """
        Initialize a VectorDB instance with the provided connection parameters.

        Args:
            connection: Direct database connection to use (optional)
            connection_pool: Existing connection pool to use (optional)
            pool_size: Number of active connections in the pool (default: 5)
            max_overflow: Maximum number of connections beyond pool_size (default: 10)
            timeout: Maximum wait time in seconds for establishing a connection (default: 30)
            **kwargs: Additional connection parameters passed to singlestoredb.connect()
                - host: Hostname, IP address, or URL for database connection
                - user: Database username
                - password: Database password
                - port: Database port
                - database: Database name
                - plus other optional parameters for customizing the connection via singlestoredb python client

        Note:
            Only one of `connection` or `connection_pool` should be provided.
            If neither is provided, a new connection pool will be created.
        """
        # Store connection arguments for pool creation
        self.connection_args = kwargs or {}
        if "conn_attrs" not in self.connection_args:
            self.connection_args["conn_attrs"] = dict()
        self.connection_args["conn_attrs"]["_connector_name"] = "vectorstore python sdk"
        self.connection_args["conn_attrs"]["_connector_version"] = metadata.version(
            "singlestore-vectorstore"
        )

        # Set up connection handling
        self.connection = connection
        self.connection_pool = None

        if not connection and not connection_pool:
            # Create a new connection pool
            self.connection_pool = QueuePool(
                creator=self._create_connection,
                pool_size=pool_size,
                max_overflow=max_overflow,
                timeout=timeout,
            )
        elif connection_pool:
            # Use provided connection pool
            self.connection_pool = connection_pool

        # Initialize the indexes table
        self._initialize_indexes_table()

    def _initialize_indexes_table(self) -> None:
        """
        Initialize the vector indexes table in the database.

        Creates the table if it doesn't already exist.
        """
        # Generate a comma-separated string of metric values for the enum
        metric_values = ", ".join([f"'{metric.value}'" for metric in Metric])

        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                # Create the indexes table with appropriate columns
                curr.execute(f"""
                    CREATE TABLE IF NOT EXISTS {INDEXES_TABLE_NAME}(
                        {INDEX_NAME_FIELD} VARCHAR(255) PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        {INDEX_DIMENSION_FIELD} INT DEFAULT {INDEX_DIMENSION_FIELD_DEFAULT},
                        {INDEX_METRIC_FIELD} Enum({metric_values}),
                        {INDEX_DELETION_PROTECT_FIELD} BOOL,
                        {INDEX_USE_VECTOR_INDEX_FIELD} BOOL,
                        {INDEX_VECTOR_INDEX_OPTIONS_FIELD} JSON,
                        {INDEX_TAGS_FIELD} JSON)
                """)
        finally:
            self._close_connection_if_needed(conn)

    def _create_connection(self) -> Connection:
        """
        Create a new database connection using the stored connection arguments.

        Returns:
            New database connection
        """
        return connect(**self.connection_args)

    def _get_connection(self) -> Connection:
        """
        Get a database connection from either the direct connection or connection pool.

        Returns:
            A database connection

        Raises:
            ValueError: If no connection or connection pool is available
        """
        if self.connection:
            return self.connection
        elif self.connection_pool:
            return self.connection_pool.connect()
        else:
            raise ValueError("No connection or connection pool available")

    def _close_connection_if_needed(self, conn: Connection) -> None:
        """
        Close the database connection if it was obtained from a pool.

        Args:
            conn: The database connection to potentially close
        """
        if not self.connection:
            conn.close()

    def _get_distance_strategy(self, metric: Union[Metric, str]) -> DistanceStrategy:
        """
        Map a similarity metric to the appropriate distance strategy.

        Args:
            metric: The similarity metric to map

        Returns:
            The corresponding distance strategy
        """
        return (
            DistanceStrategy.EUCLIDEAN_DISTANCE
            if metric == Metric.EUCLIDEAN
            else DistanceStrategy.DOT_PRODUCT
        )

    def create_index(
        self,
        name: str,
        dimension: int = 1536,
        metric: Union[Metric, str] = Metric.COSINE,
        deletion_protection: Union[
            DeletionProtection, str
        ] = DeletionProtection.DISABLED,
        tags: Optional[Dict[str, str]] = None,
        use_vector_index: bool = False,
        vector_index_options: Optional[dict] = None,
    ) -> IndexModel:
        """
        Create a new vector index in the database.

        Args:
            name: Unique name for the index
            dimension: Dimensionality of vectors to be stored (default: 1536)
            metric: Similarity metric to use (default: Metric.COSINE)
            deletion_protection: Whether to prevent accidental deletion (default: DISABLED)
            tags: Optional metadata tags for the index
            use_vector_index: Whether to create a vector index for faster searches
            vector_index_options: Configuration options for the vector index

        Returns:
            IndexModel instance representing the created index

        Raises:
            ValueError: If the metric or deletion_protection values are invalid
        """
        # Default empty collections for optional parameters
        vector_index_options = dict(vector_index_options or {})
        tags = dict(tags or {})

        # Convert string enums to their proper types if needed
        if isinstance(metric, str):
            metric = Metric(metric)
        if not isinstance(metric, Metric):
            raise ValueError(f"Invalid metric: {metric}. Must be one of {list(Metric)}")

        if isinstance(deletion_protection, str):
            deletion_protection = DeletionProtection(deletion_protection)
        if not isinstance(deletion_protection, DeletionProtection):
            raise ValueError(
                f"Invalid deletion_protection: {deletion_protection}. "
                f"Must be one of {list(DeletionProtection)}"
            )

        # Add metric type to vector index options
        vector_index_options[METRIC_TYPE] = self._get_distance_strategy(metric).value

        # Set up database connection
        connection = self._get_connection()
        try:
            with connection.cursor() as curr:
                # Insert index metadata into indexes table
                is_protected = deletion_protection == DeletionProtection.ENABLED
                curr.execute(f"""
                    INSERT INTO {INDEXES_TABLE_NAME}(
                        {INDEX_NAME_FIELD},
                        {INDEX_DIMENSION_FIELD},
                        {INDEX_METRIC_FIELD},
                        {INDEX_DELETION_PROTECT_FIELD},
                        {INDEX_USE_VECTOR_INDEX_FIELD},
                        {INDEX_VECTOR_INDEX_OPTIONS_FIELD},
                        {INDEX_TAGS_FIELD}
                    ) VALUES(
                        '{name}',
                        {dimension},
                        '{metric.value}',
                        {is_protected},
                        {use_vector_index},
                        '{json.dumps(vector_index_options)}',
                        '{json.dumps(tags)}'
                    );
                """)

                # Prepare vector index options
                index_options = f"INDEX_OPTIONS '{json.dumps(vector_index_options)}'"

                # Determine which vector field to use based on metric
                vector_field = VECTOR_FIELD

                # Build the table schema
                table_fields = f"""
                    {ID_FIELD} VARCHAR(255) PRIMARY KEY,
                    {VECTOR_FIELD} VECTOR({dimension}, F32) NOT NULL,
                    {METADATA_FIELD} JSON,
                    {NAMESPACE_FIELD} VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                """

                # Add normalized vector field for cosine similarity
                if metric == Metric.COSINE:
                    table_fields += (
                        f", {VECTOR_NORMALIZED_FIELD} VECTOR({dimension}, F32) NOT NULL"
                    )
                    vector_field = VECTOR_NORMALIZED_FIELD

                # Add vector index if requested
                if use_vector_index:
                    table_fields += f", VECTOR INDEX {VECTOR_INDEX} ({vector_field}) {index_options}"

                # Create the table
                curr.execute(
                    f"CREATE TABLE {_get_index_table_name(name)}({table_fields});"
                )
        finally:
            self._close_connection_if_needed(connection)

        # Return the model representing the created index
        return IndexModel(
            name=name,
            dimension=dimension,
            metric=metric,
            deletion_protection=deletion_protection,
            tags=tags,
            use_vector_index=use_vector_index,
            vector_index_options=vector_index_options,
        )

    def delete_index(self, name: str) -> None:
        """
        Delete an index from the database.

        Args:
            name: Name of the index to delete

        Raises:
            ValueError: If the index has deletion protection enabled
        """
        # Check if deletion is allowed
        index = self.describe_index(name)
        if index.deletion_protection == DeletionProtection.ENABLED:
            raise ValueError(
                f"Index {name} has deletion protection enabled. Cannot delete."
            )

        # Perform the deletion
        connection = self._get_connection()
        try:
            with connection.cursor() as curr:
                # Drop the index table
                curr.execute(f"DROP TABLE IF EXISTS {_get_index_table_name(name)};")

                # Remove the index metadata
                curr.execute(
                    f"DELETE FROM {INDEXES_TABLE_NAME} WHERE {INDEX_NAME_FIELD}='{name}';"
                )
        finally:
            self._close_connection_if_needed(connection)

    def list_indexes(self) -> IndexList:
        """
        List all vector indexes in the database.

        Returns:
            IndexList containing IndexModel objects for all available indexes
        """
        connection = self._get_connection()
        try:
            with connection.cursor() as curr:
                # Query all index metadata
                curr.execute(f"""
                    SELECT
                        {INDEX_NAME_FIELD},
                        {INDEX_DIMENSION_FIELD},
                        {INDEX_METRIC_FIELD},
                        {INDEX_DELETION_PROTECT_FIELD},
                        {INDEX_USE_VECTOR_INDEX_FIELD},
                        {INDEX_VECTOR_INDEX_OPTIONS_FIELD},
                        {INDEX_TAGS_FIELD}
                    FROM {INDEXES_TABLE_NAME}
                    ORDER BY {INDEX_NAME_FIELD}
                """)
                result = curr.fetchall()

                # Convert DB rows to IndexModel objects
                indexes = [
                    IndexModel(
                        name=row[0],
                        dimension=row[1],
                        metric=row[2],
                        deletion_protection=DeletionProtection.ENABLED
                        if row[3]
                        else DeletionProtection.DISABLED,
                        use_vector_index=bool(row[4]),
                        vector_index_options=row[5],
                        tags=row[6],
                    )
                    for row in result
                ]
        finally:
            self._close_connection_if_needed(connection)

        return IndexList(indexes)

    def describe_index(self, name: str) -> IndexModel:
        """
        Get detailed information about a specific index.

        Args:
            name: Name of the index to describe

        Returns:
            IndexModel instance with the index details

        Raises:
            ValueError: If the index does not exist
        """
        connection = self._get_connection()
        try:
            with connection.cursor() as curr:
                # Query the index metadata
                curr.execute(f"""
                    SELECT
                        {INDEX_NAME_FIELD},
                        {INDEX_DIMENSION_FIELD},
                        {INDEX_METRIC_FIELD},
                        {INDEX_DELETION_PROTECT_FIELD},
                        {INDEX_USE_VECTOR_INDEX_FIELD},
                        {INDEX_VECTOR_INDEX_OPTIONS_FIELD},
                        {INDEX_TAGS_FIELD}
                    FROM {INDEXES_TABLE_NAME}
                    WHERE {INDEX_NAME_FIELD}='{name}'
                """)
                result = curr.fetchone()

                # Check if the index exists
                if result is None:
                    raise ValueError(f"Index {name} does not exist.")

                # Convert DB row to IndexModel
                index = IndexModel(
                    name=result[0],
                    dimension=result[1],
                    metric=result[2],
                    deletion_protection=DeletionProtection.ENABLED
                    if result[3]
                    else DeletionProtection.DISABLED,
                    use_vector_index=bool(result[4]),
                    vector_index_options=result[5],
                    tags=result[6],
                )
        finally:
            self._close_connection_if_needed(connection)

        return index

    def has_index(self, name: str) -> bool:
        """
        Check if an index with the given name exists.

        Args:
            name: Name of the index to check

        Returns:
            True if the index exists, False otherwise
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as curr:
                curr.execute(
                    f"SELECT {INDEX_NAME_FIELD} FROM {INDEXES_TABLE_NAME} "
                    f"WHERE {INDEX_NAME_FIELD}='{name}'"
                )
                result = curr.fetchone()
                return result is not None
        finally:
            self._close_connection_if_needed(conn)

    def configure_index(
        self,
        name: str,
        deletion_protection: Optional[DeletionProtection] = None,
        tags: Optional[Dict[str, str]] = None,
        use_vector_index: Optional[bool] = None,
        vector_index_options: Optional[dict] = None,
    ) -> IndexModel:
        """
        Configure an existing index with updated parameters.

        This method allows modification of index properties like deletion protection,
        tags, vector index usage, and vector index options.

        Args:
            name: Name of the index to configure
            deletion_protection: Whether to enable deletion protection
            tags: Custom metadata tags for the index
            use_vector_index: Whether to use a vector index
            vector_index_options: Options for the vector index

        Returns:
            Updated IndexModel reflecting the changes

        Raises:
            ValueError: If the deletion_protection value is invalid
            ValueError: If the index does not exist
        """
        # Validate index exists and get current settings
        index = self.describe_index(name)

        # Build update parameters
        update_params = []

        # Handle deletion protection
        if deletion_protection is not None:
            if not isinstance(deletion_protection, DeletionProtection):
                raise ValueError(
                    f"Invalid deletion_protection: {deletion_protection}. "
                    f"Must be one of {list(DeletionProtection)}"
                )
            is_protected = deletion_protection == DeletionProtection.ENABLED
            update_params.append(f"{INDEX_DELETION_PROTECT_FIELD}={is_protected}")

        # Handle tags
        if tags is not None:
            tags_json = json.dumps(tags)
            update_params.append(f"{INDEX_TAGS_FIELD}='{tags_json}'")

        # Initialize vector index options
        vector_index_options = dict(vector_index_options or {})

        # Add vector index settings
        if use_vector_index is not None:
            update_params.append(f"{INDEX_USE_VECTOR_INDEX_FIELD}={use_vector_index}")
            # Add required metric type to options
            distance_strategy = self._get_distance_strategy(index.metric)
            vector_index_options[METRIC_TYPE] = distance_strategy.value

        # Add vector index options to update parameters if provided
        if vector_index_options:
            options_json = json.dumps(vector_index_options)
            update_params.append(f"{INDEX_VECTOR_INDEX_OPTIONS_FIELD}='{options_json}'")

        # Execute updates if there are parameters to update
        if update_params:
            connection = self._get_connection()
            try:
                with connection.cursor() as curr:
                    # Update index metadata in the indexes table
                    update_sql = (
                        f"UPDATE {INDEXES_TABLE_NAME} "
                        f"SET {', '.join(update_params)} "
                        f"WHERE {INDEX_NAME_FIELD}='{name}'"
                    )
                    curr.execute(update_sql)

                    # Handle vector index changes if requested
                    if use_vector_index is not None:
                        table_name = _get_index_table_name(name)

                        # Drop existing index if needed
                        if index.use_vector_index or not use_vector_index:
                            curr.execute(f"DROP INDEX {VECTOR_INDEX} ON {table_name}")

                        # Create new vector index if requested
                        if use_vector_index:
                            # Determine which vector field to index based on metric
                            vector_field = (
                                VECTOR_NORMALIZED_FIELD
                                if index.metric == Metric.COSINE
                                else VECTOR_FIELD
                            )

                            # Create the vector index
                            index_options = (
                                f"INDEX_OPTIONS '{json.dumps(vector_index_options)}'"
                            )
                            create_index_sql = (
                                f"ALTER TABLE {table_name} "
                                f"ADD VECTOR INDEX {VECTOR_INDEX} ({vector_field}) {index_options}"
                            )
                            curr.execute(create_index_sql)

                            # Optimize the table to apply changes
                            curr.execute(f"OPTIMIZE TABLE {table_name} FLUSH")
            finally:
                self._close_connection_if_needed(connection)

        # Return the updated index information
        return self.describe_index(name)

    def Index(self, name: str) -> IndexInterface:
        """
        Get an interface for interacting with a specific vector index.

        Args:
            name: Name of the index to access

        Returns:
            IndexInterface implementation for the specified index

        Raises:
            ValueError: If the index does not exist
        """
        # Get the index model and return an interface to it
        index_model = self.describe_index(name)
        return _Index(
            index=index_model,
            connection=self.connection,
            connection_pool=self.connection_pool,
        )
