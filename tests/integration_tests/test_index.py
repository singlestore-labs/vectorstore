from typing import Generator
import pytest

from singlestoredb.server import docker
from singlestoredb.connection import Connection
from singlestoredb import connect

from sqlalchemy.pool import Pool, QueuePool

from vectorstore import VectorDB, Metric, DeletionProtection, DistanceStrategy, IndexInterface

class TestIndex:
    @pytest.fixture(scope="class")
    def docker_server_url(self) -> Generator[str, None, None]:
        sdb = docker.start(license="")
        conn = sdb.connect()
        curr = conn.cursor()
        curr.execute("create database test_vectorstore")
        curr.close()
        conn.close()
        yield sdb.connection_url
        sdb.stop()
    
    @pytest.fixture(scope="function")
    def clean_db_url(self, docker_server_url) -> Generator[str, None, None]:
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
    def clean_db_connection(self, clean_db_url: str) -> Generator[Connection, None, None]:
        conn = connect(host=clean_db_url, database="test_vectorstore")
        yield conn
        conn.close()

    @pytest.fixture(scope="function")
    def clean_connection_params(self, clean_db_url: str) -> dict:
        return {"host" : clean_db_url, "database" : "test_vectorstore"}

    @pytest.fixture(scope="function")
    def clean_connection_pool(self, clean_db_url: str) -> Pool:
        def _create_connection() -> Connection:
            return connect(host=clean_db_url, database="test_vectorstore")
        return QueuePool(creator=_create_connection)
    
    @pytest.fixture(scope="function")
    def index_cosine(self, clean_connection_params: dict) -> IndexInterface:
        db = VectorDB(**clean_connection_params)
        db.create_index("test_index", dimension=3, metric=Metric.DOTPRODUCT)
        return db.Index("test_index")
        
    def test_upsert_1(self, index_cosine: IndexInterface, clean_connection_params) -> None:
        vectors = [
            ("id1", [0.1, 0.2, 0.3]),
            ("id2", [0.4, 0.5, 0.6]),
            ("id3", [0.7, 0.8, 0.9]),
        ]
        count = index_cosine.upsert(vectors)
        assert count == 3
        conn = connect(**clean_connection_params)
        curr = conn.cursor()
        curr.execute("select count(*) from index_test_index")
        result = curr.fetchone()
        assert result[0] == 3
        curr.close()
        conn.close()
        