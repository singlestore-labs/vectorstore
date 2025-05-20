"""
Shared fixtures for integration tests.

This module provides fixtures for setting up test environments,
database connections, and index instances used across multiple test files.
"""

from typing import Generator

import pytest
from singlestoredb import connect
from singlestoredb.connection import Connection
from singlestoredb.server import docker
from sqlalchemy.pool import Pool, QueuePool

from vectorstore import IndexInterface, Metric, VectorDB


@pytest.fixture(scope="session")
def docker_server_url() -> Generator[str, None, None]:
    """Start a SingleStore Docker server for tests."""
    sdb = docker.start(license="")
    conn = sdb.connect()
    curr = conn.cursor()
    curr.execute("create database test_vectorstore")
    curr.close()
    conn.close()
    yield sdb.connection_url
    sdb.stop()


@pytest.fixture(scope="function")
def clean_db_url(docker_server_url) -> Generator[str, None, None]:
    """Provide a clean database URL and clean up tables after test."""
    yield docker_server_url
    conn = connect(host=docker_server_url, database="test_vectorstore")
    curr = conn.cursor()
    curr.execute("show tables")
    results = curr.fetchall()
    for result in results:
        curr.execute(f"drop table {result[0]}")
    curr.close()
    conn.close()


@pytest.fixture(scope="function")
def clean_db_connection(clean_db_url: str) -> Generator[Connection, None, None]:
    """Provide a database connection."""
    conn = connect(host=clean_db_url, database="test_vectorstore")
    yield conn
    conn.close()


@pytest.fixture(scope="function")
def clean_connection_params(clean_db_url: str) -> dict:
    """Provide connection parameters."""
    return {"host": clean_db_url, "database": "test_vectorstore"}


@pytest.fixture(scope="function")
def clean_connection_pool(clean_db_url: str) -> Pool:
    """Provide a connection pool."""

    def _create_connection() -> Connection:
        return connect(host=clean_db_url, database="test_vectorstore")

    return QueuePool(creator=_create_connection)


@pytest.fixture(
    params=[Metric.DOTPRODUCT, Metric.COSINE, Metric.EUCLIDEAN], scope="function"
)
def index_metric(
    request, clean_connection_params: dict
) -> tuple[IndexInterface, Metric]:
    """Create an index with the specified metric."""
    metric = request.param
    db = VectorDB(**clean_connection_params)
    db.create_index("test_index", dimension=3, metric=metric)
    return db.Index("test_index"), metric


@pytest.fixture(scope="function")
def index(index_metric: tuple[IndexInterface, Metric]) -> IndexInterface:
    """Get the index from index_metric."""
    return index_metric[0]


@pytest.fixture(scope="function")
def index_with_sample_data(index: IndexInterface) -> IndexInterface:
    """Create an index with some sample data."""
    vectors = [
        ("id1", [0.1, 0.2, 0.3], {"key1": "value1"}),
        ("id2", [0.4, 0.5, 0.6], {"key2": "value2"}),
        ("id3", [0.7, 0.8, 0.9], {"key3": "value3"}),
    ]
    index.upsert(vectors, namespace="test_namespace")
    return index


@pytest.fixture(scope="function")
def index_dotproduct(clean_connection_params: dict) -> IndexInterface:
    """Create an index with dot product metric."""
    db = VectorDB(**clean_connection_params)
    db.create_index("test_index_dotproduct", dimension=3, metric=Metric.DOTPRODUCT)
    return db.Index("test_index_dotproduct")


@pytest.fixture(scope="function")
def index_cosine(clean_connection_params: dict) -> IndexInterface:
    """Create an index with cosine similarity metric."""
    db = VectorDB(**clean_connection_params)
    db.create_index("test_index_cosine", dimension=3, metric=Metric.COSINE)
    return db.Index("test_index_cosine")


@pytest.fixture(scope="function")
def index_euclidean(clean_connection_params: dict) -> IndexInterface:
    """Create an index with Euclidean distance metric."""
    db = VectorDB(**clean_connection_params)
    db.create_index("test_index_euclidean", dimension=3, metric=Metric.EUCLIDEAN)
    return db.Index("test_index_euclidean")


@pytest.fixture(scope="function")
def cosine_index_with_vector_index(clean_connection_params) -> IndexInterface:
    """Create a cosine similarity index with vector indexing."""
    db = VectorDB(**clean_connection_params)
    db.create_index(
        "test_index_cosine_with_vector_index",
        dimension=3,
        metric=Metric.COSINE,
        use_vector_index=True,
        vector_index_options={"index_type": "IVF_PQFS", "nlist": 1024, "nprobe": 20},
    )
    return db.Index("test_index_cosine_with_vector_index")


@pytest.fixture(scope="function")
def dotproduct_index_with_vector_index(clean_connection_params) -> IndexInterface:
    """Create a dot product index with vector indexing."""
    db = VectorDB(**clean_connection_params)
    db.create_index(
        "test_index_dotproduct_with_vector_index",
        dimension=3,
        metric=Metric.DOTPRODUCT,
        use_vector_index=True,
        vector_index_options={"index_type": "HNSW_FLAT", "M": 16},
    )
    return db.Index("test_index_dotproduct_with_vector_index")


@pytest.fixture(scope="function")
def euclidean_index_with_vector_index(clean_connection_params) -> IndexInterface:
    """Create a Euclidean distance index with vector indexing."""
    db = VectorDB(**clean_connection_params)
    db.create_index(
        "test_index_euclidean_with_vector_index",
        dimension=3,
        metric=Metric.EUCLIDEAN,
        use_vector_index=True,
        vector_index_options={"index_type": "HNSW_PQ", "nbits": 12, "m": 3},
    )
    return db.Index("test_index_euclidean_with_vector_index")
