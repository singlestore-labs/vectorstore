import json
from abc import ABC
from typing import Dict, Optional, Union

# Third-party imports
from singlestoredb import connect
from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool, QueuePool

from .delete_protection import DeletionProtection
from .distance_strategy import DistanceStrategy
from .index_list import IndexList

# Local imports
from .index_model import IndexModel
from .metric import Metric
from ._index import _Index
from .index_interface import IndexInterface

from ._constants import (
    ID_FIELD,
    VECTOR_FIELD,
    METADATA_FIELD,
    VECTOR_NORMALIZED_FIELD,
    NAMESPACE_FIELD,
    VECTOR_INDEX,
    INDEXES_TABLE_NAME,
    INDEX_NAME_FIELD,
    INDEX_DIMENSION_FIELD,
    INDEX_DIMENSION_FIELD_DEFAULT,
    INDEX_METRIC_FIELD,
    INDEX_DELETION_PROTECT_FIELD,
    INDEX_USE_VECTOR_INDEX_FIELD,
    INDEX_VECTOR_INDEX_OPTIONS_FIELD,
    INDEX_TAGS_FIELD,
    _get_index_table_name,
    METRIC_TYPE,
)


class VectorDB(ABC):

    def _create_connection(self) -> Connection:
        """Create a new connection to the database."""
        return connect(**self.connection_args)

    def _get_connection(self) -> Connection:
        """Get a connection from the pool."""
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

    def _get_distance_strategy(self, metric: Union[Metric, str]) -> DistanceStrategy:
        return DistanceStrategy.EUCLIDEAN_DISTANCE if metric == Metric.EUCLIDEAN else DistanceStrategy.DOT_PRODUCT

    def __init__(
        self,
        *,
        connection: Optional[Connection] = None,
        connection_pool: Optional[Pool] = None,
        pool_size: Optional[int] = 5,
        max_overflow: Optional[int] = 10,
        timeout: Optional[float] = 30,
        **kwargs,
    ):
        """
        Following arguments pertain to the connection pool:

            pool_size (int, optional): Determines the number of active connections in
                the pool. Defaults to 5.
            max_overflow (int, optional): Determines the maximum number of connections
                allowed beyond the pool_size. Defaults to 10.
            timeout (float, optional): Specifies the maximum wait time in seconds for
                establishing a connection. Defaults to 30.

            Following arguments pertain to the database connection:

            host (str, optional): Specifies the hostname, IP address, or URL for the
                database connection. The default scheme is "mysql".
            user (str, optional): Database username.
            password (str, optional): Database password.
            port (int, optional): Database port. Defaults to 3306 for non-HTTP
                connections, 80 for HTTP connections, and 443 for HTTPS connections.
            database (str, optional): Database name.

            Additional optional arguments provide further customization over the
            database connection:

            pure_python (bool, optional): Toggles the connector mode. If True,
                operates in pure Python mode.
            local_infile (bool, optional): Allows local file uploads.
            charset (str, optional): Specifies the character set for string values.
            ssl_key (str, optional): Specifies the path of the file containing the SSL
                key.
            ssl_cert (str, optional): Specifies the path of the file containing the SSL
                certificate.
            ssl_ca (str, optional): Specifies the path of the file containing the SSL
                certificate authority.
            ssl_cipher (str, optional): Sets the SSL cipher list.
            ssl_disabled (bool, optional): Disables SSL usage.
            ssl_verify_cert (bool, optional): Verifies the server's certificate.
                Automatically enabled if ``ssl_ca`` is specified.
            ssl_verify_identity (bool, optional): Verifies the server's identity.
            conv (dict[int, Callable], optional): A dictionary of data conversion
                functions.
            credential_type (str, optional): Specifies the type of authentication to
                use: auth.PASSWORD, auth.JWT, or auth.BROWSER_SSO.
            autocommit (bool, optional): Enables autocommits.
            results_type (str, optional): Determines the structure of the query results:
                tuples, namedtuples, dicts.
            results_format (str, optional): Deprecated. This option has been renamed to
                results_type.
        """
        self.connection = None
        self.connection_args = kwargs
        if connection:
            self.connection = connection
        elif connection_pool:
            self.connection_pool = connection_pool
        else:
            self.connection_pool = QueuePool(
                creator=self._create_connection,
                pool_size=pool_size,
                max_overflow=max_overflow,
                timeout=timeout,
            )
        metric_values = ", ".join(["'" + metric.value +"'" for metric in Metric])
        conn = self._get_connection()
        try:
            curr = conn.cursor()
            try:
                curr.execute(f"""CREATE TABLE IF NOT EXISTS {INDEXES_TABLE_NAME}(
                              {INDEX_NAME_FIELD} VARCHAR(255) PRIMARY KEY,
                              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                              {INDEX_DIMENSION_FIELD} INT DEFAULT {INDEX_DIMENSION_FIELD_DEFAULT},
                              {INDEX_METRIC_FIELD} Enum({metric_values}),
                              {INDEX_DELETION_PROTECT_FIELD} bool,
                              {INDEX_USE_VECTOR_INDEX_FIELD} bool,
                              {INDEX_VECTOR_INDEX_OPTIONS_FIELD} JSON,
                              {INDEX_TAGS_FIELD} JSON)
                              """)
            finally:
                curr.close()
        finally:
            self._close_connection_if_needed(conn)

    def create_index(
        self,
        name: str,
        dimension: Optional[int]=1536,
        metric: Optional[Union[Metric, str]] = Metric.COSINE,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        tags: Optional[Dict[str, str]] = None,
        use_vector_index: bool = False,
        vector_index_options: Optional[dict] = None,
    ) -> IndexModel:
        connection = self._get_connection()
        try:
            curr = connection.cursor()
            try:
                vector_index_options = dict(vector_index_options or {})
                tags = dict(tags or {})
                if isinstance(metric, str):
                    metric = Metric(metric)
                if not isinstance(metric, Metric):
                    raise ValueError(f"Invalid metric: {metric}. Must be one of {list(Metric)}")
                if isinstance(deletion_protection, str):
                    deletion_protection = DeletionProtection(deletion_protection)
                if not isinstance(deletion_protection, DeletionProtection):
                    raise ValueError(f"Invalid deletion_protection: {deletion_protection}. Must be one of {list(DeletionProtection)}")
                vector_index_options[METRIC_TYPE] = self._get_distance_strategy(metric).value
                curr.execute(f"""INSERT INTO {INDEXES_TABLE_NAME}(
                            {INDEX_NAME_FIELD},
                            {INDEX_DIMENSION_FIELD},
                            {INDEX_METRIC_FIELD},
                            {INDEX_DELETION_PROTECT_FIELD},
                            {INDEX_USE_VECTOR_INDEX_FIELD},
                            {INDEX_VECTOR_INDEX_OPTIONS_FIELD},
                            {INDEX_TAGS_FIELD}) VALUES(
                             '{name}',
                             {dimension},
                             '{metric.value}',
                             {deletion_protection == DeletionProtection.ENABLED},
                             {use_vector_index},
                             '{json.dumps(vector_index_options)}',
                             '{json.dumps(tags)}');""")
                index_options = f"INDEX_OPTIONS '{json.dumps(vector_index_options)}'"
                vector_field = VECTOR_FIELD
                table_fields = f"""{ID_FIELD} VARCHAR(255) PRIMARY KEY,
                {VECTOR_FIELD} VECTOR({dimension}, F32) NOT NULL,
                {METADATA_FIELD} JSON,
                {NAMESPACE_FIELD} VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"""
                if metric == Metric.COSINE:
                    table_fields += f", {VECTOR_NORMALIZED_FIELD} VECTOR({dimension}, F32) NOT NULL"
                    vector_field = VECTOR_NORMALIZED_FIELD
                if use_vector_index:
                    table_fields += f", VECTOR INDEX {VECTOR_INDEX} ({vector_field}) {index_options}"
                curr.execute(f"""CREATE TABLE {_get_index_table_name(name)}({table_fields});""")
            finally:
                curr.close()
        finally:
            self._close_connection_if_needed(connection)

        return IndexModel(
            name=name,
            dimension=dimension,
            metric=metric,
            deletion_protection=deletion_protection,
            tags=tags if tags is not None else {},
            use_vector_index=use_vector_index,
            vector_index_options=vector_index_options
        )
        

    def delete_index(self, name: str) -> None:
        """Delete an index from the database."""
        index = self.describe_index(name)
        if index.deletion_protection == DeletionProtection.ENABLED:
            raise ValueError(f"Index {name} has deletion protection enabled. Cannot delete.")
        connection = self._get_connection()
        try:
            curr = connection.cursor()
            try:
                curr.execute(f"DROP TABLE IF EXISTS {_get_index_table_name(name)};")
                curr.execute(f"DELETE FROM {INDEXES_TABLE_NAME} WHERE {INDEX_NAME_FIELD}='{name}';")
            finally:
                curr.close()
        finally:
            self._close_connection_if_needed(connection)

    def list_indexes(self) -> IndexList:
        connection = self._get_connection()
        try:
            curr = connection.cursor()
            try:
                curr.execute(f"""SELECT {INDEX_NAME_FIELD},
                             {INDEX_DIMENSION_FIELD},
                             {INDEX_METRIC_FIELD},
                             {INDEX_DELETION_PROTECT_FIELD},
                             {INDEX_TAGS_FIELD},
                             {INDEX_VECTOR_INDEX_OPTIONS_FIELD},
                             {INDEX_TAGS_FIELD} FROM {INDEXES_TABLE_NAME}
                             order by {INDEX_NAME_FIELD}""")
                result = curr.fetchall()
                indexes = [IndexModel(
                    name=row[0],
                    dimension=row[1],
                    metric=row[2],
                    deletion_protection=DeletionProtection.ENABLED if row[3] else DeletionProtection.DISABLED,
                    use_vector_index=bool(row[4]),
                    vector_index_options=row[5],
                    tags=row[6]) for row in result]
            finally:
                curr.close()
        finally:
            self._close_connection_if_needed(connection)
        return IndexList(indexes)

    def describe_index(self, name: str) -> IndexModel:
        connection = self._get_connection()
        try:
            curr = connection.cursor()
            try:
                curr.execute(f"""SELECT {INDEX_NAME_FIELD},
                             {INDEX_DIMENSION_FIELD},
                             {INDEX_METRIC_FIELD},
                             {INDEX_DELETION_PROTECT_FIELD},
                             {INDEX_USE_VECTOR_INDEX_FIELD},
                             {INDEX_VECTOR_INDEX_OPTIONS_FIELD},
                             {INDEX_TAGS_FIELD} FROM {INDEXES_TABLE_NAME} 
                             WHERE {INDEX_NAME_FIELD}='{name}'""")
                result = curr.fetchone()
                if result is None:
                    raise ValueError(f"Index {name} does not exist.")
                index = IndexModel(
                    name=result[0],
                    dimension=result[1],
                    metric=result[2],
                    deletion_protection=DeletionProtection.ENABLED if result[3] else DeletionProtection.DISABLED,
                    use_vector_index=bool(result[4]),
                    vector_index_options=result[5],
                    tags=result[6],
                )
            finally:
                curr.close()
        finally:
            self._close_connection_if_needed(connection)
        return index

    def has_index(self, name: str) -> bool:
        conn = self._get_connection()
        is_exists = False
        try:
            curr = conn.cursor()
            try:
                curr.execute(f"SELECT {INDEX_NAME_FIELD} FROM {INDEXES_TABLE_NAME} WHERE {INDEX_NAME_FIELD}='{name}'")
                result = curr.fetchone()
                is_exists = result is not None
            finally:
                curr.close()
        finally:
            self._close_connection_if_needed(conn)
        return is_exists

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
        
        Args:
            name: Name of the index to configure
            deletion_protection: Whether to enable deletion protection
            tags: Custom metadata tags for the index
            use_vector_index: Whether to use a vector index
            vector_index_options: Options for the vector index
            
        Returns:
            Updated IndexModel
        """
        # Validate index exists and get current settings
        index = self.describe_index(name)
        
        # Build update parameters
        update_params = []
        
        # Handle deletion protection
        if deletion_protection is not None:
            if not isinstance(deletion_protection, DeletionProtection):
                raise ValueError(f"Invalid deletion_protection: {deletion_protection}. Must be one of {list(DeletionProtection)}")
            is_protected = deletion_protection == DeletionProtection.ENABLED
            update_params.append(f"{INDEX_DELETION_PROTECT_FIELD}={is_protected}")
        
        # Handle tags
        if tags is not None:
            tags_json = json.dumps(tags)
            update_params.append(f"{INDEX_TAGS_FIELD}='{tags_json}'")
        
        # Prepare vector index options
        if vector_index_options is None:
            vector_index_options = {}
        
        # Handle vector index settings
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
                    update_sql = f"UPDATE {INDEXES_TABLE_NAME} SET {', '.join(update_params)} WHERE {INDEX_NAME_FIELD}='{name}'"
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
                            vector_field = VECTOR_NORMALIZED_FIELD if index.metric == Metric.COSINE else VECTOR_FIELD
                            
                            # Create the vector index
                            index_options = f"INDEX_OPTIONS '{json.dumps(vector_index_options)}'"
                            create_index_sql = f"ALTER TABLE {table_name} ADD VECTOR INDEX {VECTOR_INDEX} ({vector_field}) {index_options}"
                            curr.execute(create_index_sql)
                            
                            # Optimize the table to apply changes
                            curr.execute(f"OPTIMIZE TABLE {table_name} FLUSH")
            finally:
                self._close_connection_if_needed(connection)
                
        # Return the updated index information
        return self.describe_index(name)

    def Index(self, name: str = "") -> IndexInterface:
        return _Index(index=self.describe_index(name), connection=self.connection, connection_pool=self.connection_pool)