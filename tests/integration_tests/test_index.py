from typing import Generator, List
import pytest

from singlestoredb.server import docker
from singlestoredb.connection import Connection
from singlestoredb import connect

from sqlalchemy.pool import Pool, QueuePool

from vectorstore import VectorDB, Metric, DeletionProtection, DistanceStrategy, IndexInterface
from vectorstore.match import MatchTypedDict
from vectorstore.vector import Vector

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

    @pytest.fixture(params=[Metric.DOTPRODUCT, Metric.COSINE, Metric.EUCLIDEAN], scope="function")
    def index_metric(self, request, clean_connection_params: dict) -> tuple[IndexInterface, Metric]:
        metric = request.param
        db = VectorDB(**clean_connection_params)
        db.create_index("test_index", dimension=3, metric=metric)
        return db.Index("test_index"), metric

    @pytest.fixture(scope="function")
    def index(self, index_metric: tuple[IndexInterface, Metric]) -> IndexInterface:
        return index_metric[0]

    @pytest.fixture(scope="function")
    def index_with_sample_data(self, index: IndexInterface) -> IndexInterface:
        vectors = [
            ("id1", [0.1, 0.2, 0.3], {"key1": "value1"}),
            ("id2", [0.4, 0.5, 0.6], {"key2": "value2"}),
            ("id3", [0.7, 0.8, 0.9], {"key3": "value3"}),
        ]
        index.upsert(vectors, namespace="test_namespace")
        return index

    @pytest.fixture(scope="function")
    def index_dotproduct(self, clean_connection_params: dict) -> IndexInterface:
        db = VectorDB(**clean_connection_params)
        db.create_index("test_index", dimension=3, metric=Metric.DOTPRODUCT)
        return db.Index("test_index")

    @pytest.fixture(scope="function")
    def index_cosine(self, clean_connection_params: dict) -> IndexInterface:
        db = VectorDB(**clean_connection_params)
        db.create_index("test_index", dimension=3, metric=Metric.COSINE)
        return db.Index("test_index")

    @pytest.fixture(scope="function")
    def index_euclidean(self, clean_connection_params: dict) -> IndexInterface:
        db = VectorDB(**clean_connection_params)
        db.create_index("test_index", dimension=3, metric=Metric.EUCLIDEAN)
        return db.Index("test_index")

    def test_upsert(self, clean_connection_params, index_metric: tuple[IndexInterface, Metric]) -> None:
        vectors = [
            ("id1", [0.1, 0.2, 0.3]),
            ("id2", [0.4, 0.5, 0.6]),
            ("id3", [0.7, 0.8, 0.9]),
        ]
        index = index_metric[0]
        metric = index_metric[1]
        count = index.upsert(vectors)
        assert count == 3
        conn = connect(**clean_connection_params)
        curr = conn.cursor()
        curr.execute("select count(*) from index_test_index")
        result = curr.fetchone()
        assert result[0] == 3
        curr.execute("select * from index_test_index order by id")
        results = curr.fetchall()
        assert len(results) == 3
        for i, vector in enumerate(vectors):
            if metric == Metric.COSINE:
                assert len(results[i]) == 7
                length = sum(vector[1][j]**2 for j in range(3))**0.5
                assert results[i][6] == pytest.approx(
                    [vector[1][j]/length for j in range(3)], rel=1e-3
                )
            else:
                assert len(results[i]) == 6
            assert results[i][0] == vector[0]
            assert results[i][1] == pytest.approx(vector[1], rel=1e-3)
            assert results[i][2] is None
            assert results[i][3] is None
        curr.close()
        conn.close()

    def test_upsert_with_metadata(self, clean_connection_params: dict, index: IndexInterface):
        vectors = [
            ("id1", [0.1, 0.2, 0.3], {"key1": "value1"}),
            ("id2", [0.4, 0.5, 0.6], {"key2": "value2"}),
            ("id3", [0.7, 0.8, 0.9], {"key3": "value3"}),
        ]
        count = index.upsert(vectors)
        assert count == 3
        conn = connect(**clean_connection_params)
        curr = conn.cursor()
        curr.execute("select count(*) from index_test_index")
        result = curr.fetchone()
        assert result[0] == 3
        curr.execute("select * from index_test_index order by id")
        results = curr.fetchall()
        assert len(results) == 3
        for i, vector in enumerate(vectors):
            assert len(results[i]) >= 6
            assert results[i][0] == vector[0]
            assert results[i][1] == pytest.approx(vector[1], rel=1e-3)
            assert results[i][2] == vector[2]
            assert results[i][3] is None
            if len(results[i]) > 6:
                assert len(results[i]) == 7
                length = sum(vector[1][j]**2 for j in range(3))**0.5
                assert results[i][6] == pytest.approx(
                    [vector[1][j]/length for j in range(3)], rel=1e-3
                )
        curr.close()
        conn.close()

    def test_upsert_with_namespace(self, clean_connection_params: dict, index: IndexInterface):
        vectors = [
            ("id1", [0.1, 0.2, 0.3], {"key1": "value1"}),
            ("id2", [0.4, 0.5, 0.6], {"key2": "value2"}),
            ("id3", [0.7, 0.8, 0.9], {"key3": "value3"}),
        ]
        count = index.upsert(vectors, namespace="test_namespace")
        assert count == 3
        conn = connect(**clean_connection_params)
        curr = conn.cursor()
        curr.execute("select count(*) from index_test_index")
        result = curr.fetchone()
        assert result[0] == 3
        curr.execute("select * from index_test_index order by id")
        results = curr.fetchall()
        assert len(results) == 3
        for i, vector in enumerate(vectors):
            assert len(results[i]) >= 6
            assert results[i][0] == vector[0]
            assert results[i][1] == pytest.approx(vector[1], rel=1e-3)
            assert results[i][2] == vector[2]
            assert results[i][3] == "test_namespace"
            if len(results[i]) > 6:
                assert len(results[i]) == 7
                length = sum(vector[1][j]**2 for j in range(3))**0.5
                assert results[i][6] == pytest.approx(
                    [vector[1][j]/length for j in range(3)], rel=1e-3
                )
        curr.close()
        conn.close()

    def test_upset_vector_objects(self, clean_connection_params: dict, index: IndexInterface):
        vectors = [
            Vector(id="id1", vector=[0.1, 0.2, 0.3], metadata={"key1": "value1"}),
            Vector(id="id2", vector=[0.5, 0.6, 0.7], metadata={"key2": "value2"}),
            Vector(id="id3", vector=[0.8, 0.9, 0.10], metadata={"key3": "value3"}),
        ]
        count = index.upsert(vectors, namespace="test_namespace")
        assert count == 3
        conn = connect(**clean_connection_params)
        curr = conn.cursor()
        curr.execute("select count(*) from index_test_index")
        result = curr.fetchone()
        assert result[0] == 3
        curr.execute("select * from index_test_index order by id")
        results = curr.fetchall()
        assert len(results) == 3
        for i, vector in enumerate(vectors):
            assert len(results[i]) >= 6
            assert results[i][0] == vector.id
            assert results[i][1] == pytest.approx(vector.vector, rel=1e-3)
            assert results[i][2] == vector.metadata
            assert results[i][3] == "test_namespace"
            if len(results[i]) > 6:
                assert len(results[i]) == 7
                length = sum(vector.vector[j]**2 for j in range(3))**0.5
                assert results[i][6] == pytest.approx(
                    [vector.vector[j]/length for j in range(3)], rel=1e-3
                )
        curr.close()
        conn.close()

    def test_upsert_named_dicts(self, clean_connection_params: dict, index: IndexInterface):
        vectors = [
            {"id": "id1", "values": [0.1, 0.2, 0.3], "metadata": {"key1": "value1"}},
            {"id": "id2", "values": [0.4, 0.5, 0.6], "metadata": {"key2": "value2"}},
            {"id": "id3", "values": [0.7, 0.8, 0.9], "metadata": {"key3": "value3"}},
        ]
        count = index.upsert(vectors)
        assert count == 3
        conn = connect(**clean_connection_params)
        curr = conn.cursor()
        curr.execute("select count(*) from index_test_index")
        result = curr.fetchone()
        assert result[0] == 3
        curr.execute("select * from index_test_index order by id")
        results = curr.fetchall()
        assert len(results) == 3
        for i, vector in enumerate(vectors):
            assert len(results[i]) >= 6
            assert results[i][0] == vector["id"]
            assert results[i][1] == pytest.approx(vector["values"], rel=1e-3)
            assert results[i][2] == vector["metadata"]
            assert results[i][3] is None
            if len(results[i]) > 6:
                assert len(results[i]) == 7
                length = sum(vector["values"][j]**2 for j in range(3))**0.5
                assert results[i][6] == pytest.approx(
                    [vector["values"][j]/length for j in range(3)], rel=1e-3
                )
        curr.close()
        conn.close()

    def test_upsert_from_df(self, clean_connection_params: dict, index: IndexInterface):
        import pandas as pd
        vectors = pd.DataFrame({
            "id": ["id1", "id2", "id3", "id2"],
            "values": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.1, 0.11, 0.12]],
            "metadata": [{"key1": "value1"}, {"key2": "value2"}, {"key3": "value3"}, {"key4": "value4"}]
        })
        count = index.upsert_from_dataframe(vectors, namespace="test_namespace")
        assert count == 5
        conn = connect(**clean_connection_params)
        curr = conn.cursor()
        curr.execute("select count(*) from index_test_index")
        result = curr.fetchone()
        assert result[0] == 3
        curr.execute("select * from index_test_index order by id")
        results = curr.fetchall()
        assert len(results) == 3
        for i, vector in enumerate(vectors.values):
            if i == 1:
                continue
            if i == 3:
                i = 1
            assert len(results[i]) >= 6
            assert results[i][0] == vector[0]
            assert results[i][1] == pytest.approx(vector[1], rel=1e-3)
            assert results[i][2] == vector[2]
            assert results[i][3] == "test_namespace"
            if len(results[i]) > 6:
                assert len(results[i]) == 7
                assert len(results[i][6]) == 3
                length = sum(vector[1][j]**2 for j in range(3))**0.5
                assert results[i][6] == pytest.approx(
                    [vector[1][j]/length for j in range(3)], rel=1e-3
                )
        curr.close()
        conn.close()

    def test_list(self, index: IndexInterface) -> None:
        index.upsert([
            ("id1", [0.1, 0.2, 0.3]),
            ("pid2", [0.4, 0.5, 0.6]),
            ("id3", [0.7, 0.8, 0.9]),
        ], namespace="namespace1")
        index.upsert([
            ("pid4", [0.1, 0.2, 0.3]),
            ("id5", [0.4, 0.5, 0.6]),
            ("pid6", [0.7, 0.8, 0.9]),
        ], namespace="namespace2")
        all_list = index.list()
        assert all_list == ["id1", "id3",  "id5", "pid2", "pid4", "pid6"]
        namespace1_list = index.list(namespace="namespace1")
        assert namespace1_list == ["id1", "id3", "pid2"]
        namespace2_list = index.list(namespace="namespace2")
        assert namespace2_list == ["id5", "pid4", "pid6"]
        empty_list = index.list(namespace="namespace3")
        assert empty_list == []
        empty_list_prefix = index.list(prefix="DD")
        assert empty_list_prefix == []
        prefix_list1 = index.list(prefix="id")
        assert prefix_list1 == ["id1", "id3", "id5"]
        prefix_list2 = index.list(prefix="pid")
        assert prefix_list2 == ["pid2", "pid4", "pid6"]
        prefix_namespace_list = index.list(prefix="id", namespace="namespace1")
        assert prefix_namespace_list == ["id1", "id3"]

    def test_fetch_1(self, index_with_sample_data: IndexInterface) -> None:
        fetched_vectors = index_with_sample_data.fetch(["id1", "id2"])
        # Check the structure of the result
        assert set(fetched_vectors.keys()) == {"id1", "id2"}
        # Check id1 vector properties
        assert fetched_vectors["id1"].id == "id1"
        assert fetched_vectors["id1"].metadata == {"key1": "value1"}
        assert fetched_vectors["id1"].vector == pytest.approx([0.1, 0.2, 0.3], rel=1e-3)
        # Check id2 vector properties
        assert fetched_vectors["id2"].id == "id2"
        assert fetched_vectors["id2"].metadata == {"key2": "value2"}
        assert fetched_vectors["id2"].vector == pytest.approx([0.4, 0.5, 0.6], rel=1e-3)

    def test_fetch_2(self, index_with_sample_data: IndexInterface) -> None:
        # Test non-existent vector
        empty_fetch = index_with_sample_data.fetch(["nonexistent_id"])
        assert empty_fetch == {}

    def test_fetch_3(self, index_with_sample_data: IndexInterface) -> None:
        # Test with non-existent namespace
        empty_fetch_namespace = index_with_sample_data.fetch(["id1"], namespace="nonexistent_namespace")
        assert empty_fetch_namespace == {}

    def test_fetch_4(self, index_with_sample_data: IndexInterface) -> None:
        # Test with empty list
        namespace_fetch = index_with_sample_data.fetch(None, namespace="test_namespace")
        assert len(namespace_fetch) == 3

    def test_fetch_4(self, index_with_sample_data: IndexInterface) -> None:
        # Test with correct namespace
        fetch_namespace = index_with_sample_data.fetch(["id1"], namespace="test_namespace")
        assert len(fetch_namespace) == 1
        assert "id1" in fetch_namespace
        assert fetch_namespace["id1"].id == "id1"
        assert fetch_namespace["id1"].metadata == {"key1": "value1"}
        assert fetch_namespace["id1"].vector == pytest.approx([0.1, 0.2, 0.3], rel=1e-3)

    def test_update_1(self, index_with_sample_data: IndexInterface) -> None:
        # Update existing vector
        result = index_with_sample_data.update("id1", values = [0.9, 0.8, 0.7])
        assert result == {}
        fetched_vectors = index_with_sample_data.fetch(["id1"])
        assert fetched_vectors["id1"].id == "id1"
        assert fetched_vectors["id1"].metadata == {"key1": "value1"}
        assert fetched_vectors["id1"].vector == pytest.approx([0.9, 0.8, 0.7], rel=1e-3)

    def test_update_2(self, index_with_sample_data: IndexInterface) -> None:
        # Update non-existent vector
        result = index_with_sample_data.update("nonexistent_id", values=[0.1, 0.2, 0.3])
        assert result == {}
        assert index_with_sample_data.fetch(["nonexistent_id"]) == {}

    def test_update_3(self, index_with_sample_data: IndexInterface) -> None:
        # Update vector with metadata
        result = index_with_sample_data.update("id1", values=[0.9, 0.8, 0.7], set_metadata={"key1": "new_value"})
        assert result == {}
        fetched_vectors = index_with_sample_data.fetch(["id1"])
        assert fetched_vectors["id1"].id == "id1"
        assert fetched_vectors["id1"].metadata == {"key1": "new_value"}
        assert fetched_vectors["id1"].vector == pytest.approx([0.9, 0.8, 0.7], rel=1e-3)

    def test_update_4(self, index_with_sample_data: IndexInterface) -> None:
        # Update vector with namespace
        result = index_with_sample_data.update("id1", set_metadata={}, namespace="test_namespace")
        assert result == {}
        fetched_vectors = index_with_sample_data.fetch(["id1"], namespace="test_namespace")
        assert len(fetched_vectors) == 1
        assert fetched_vectors["id1"].id == "id1"
        assert fetched_vectors["id1"].metadata == {}
        assert fetched_vectors["id1"].vector == pytest.approx([0.1, 0.2, 0.3], rel=1e-3)

    def test_update_5(self, index_with_sample_data: IndexInterface) -> None:
        # Error when trying to update vector without providing values
        with pytest.raises(ValueError):
            index_with_sample_data.update("id1")

    def test_update_6(self, index_with_sample_data: IndexInterface) -> None:
        # Error when trying to update vector with empty values
        with pytest.raises(ValueError):
            index_with_sample_data.update(None, values=[0.1, 0.2, 0.3])

    def test_describe_index_stats_1(self, index_with_sample_data: IndexInterface) -> None:
        stats = index_with_sample_data.describe_index_stats()
        assert stats == {
            "dimension": 3,
            "total_vector_count": 3,
            "namespaces": {"test_namespace": {"vector_count": 3}}
        }

    def test_describe_index_stats_with_filter(self, index_with_sample_data: IndexInterface) -> None:
        pass

    def test_delete_1(self, index_with_sample_data: IndexInterface) -> None:
        # Delete existing vector
        index_with_sample_data.delete(["id1"])
        fetched_vectors = index_with_sample_data.fetch(["id1"])
        assert fetched_vectors == {}
        assert index_with_sample_data.list() == ["id2", "id3"]

    def test_delete_2(self, index_with_sample_data: IndexInterface) -> None:
        # Delete non-existent vector
        index_with_sample_data.delete(["nonexistent_id"])
        fetched_vectors = index_with_sample_data.fetch(["nonexistent_id"])
        assert fetched_vectors == {}
        assert index_with_sample_data.list() == ["id1", "id2", "id3"]

    def test_delete_3(self, index_with_sample_data: IndexInterface) -> None:
        # Delete vector with namespace
        index_with_sample_data.delete(["id1"], namespace="test_namespace")
        fetched_vectors = index_with_sample_data.fetch(["id1"], namespace="test_namespace")
        assert fetched_vectors == {}
        assert index_with_sample_data.list(namespace="test_namespace") == ["id2", "id3"]

    def test_delete_4(self, index_with_sample_data: IndexInterface) -> None:
        # Delete all vectors in a namespace
        index_with_sample_data.delete(namespace="test_namespace", delete_all=True)
        fetched_vectors = index_with_sample_data.fetch(["id1", "id2", "id3"], namespace="test_namespace")
        assert fetched_vectors == {}
        assert index_with_sample_data.list(namespace="test_namespace") == []

    def test_delete_5(self, index_with_sample_data: IndexInterface) -> None:
        # Delete all vectors in the index
        index_with_sample_data.delete([], delete_all=True)
        fetched_vectors = index_with_sample_data.fetch(["id1", "id2", "id3"])
        assert fetched_vectors == {}
        assert index_with_sample_data.list() == []

    def test_delete_6(self, index_with_sample_data: IndexInterface) -> None:
        # Error when trying to delete without providing IDs or namespace
        with pytest.raises(ValueError):
            index_with_sample_data.delete()

    def test_delete_7(self, index_with_sample_data: IndexInterface) -> None:
        # Error when trying to delete with empty IDs
        with pytest.raises(ValueError):
            index_with_sample_data.delete([])

    def test_delete_8(self, index_with_sample_data: IndexInterface) -> None:
        # Error when trying to delete with empty namespace
        with pytest.raises(ValueError):
            index_with_sample_data.delete(namespace="mynamespace")

    def test_delete_9(self, index_with_sample_data: IndexInterface) -> None:
        # Error when trying to delete with empty IDs and namespace
        with pytest.raises(ValueError):
            index_with_sample_data.delete([], namespace="mynamespace")

    def test_delete_10(self, index_with_sample_data: IndexInterface) -> None:
        # Error when trying to delete with empty IDs and delete_all=True
        with pytest.raises(ValueError):
            index_with_sample_data.delete(["id1"], delete_all=True)

    def test_delete_11(self, index_with_sample_data: IndexInterface) -> None:
        # Error when trying to delete with empty namespace and delete_all=True
        with pytest.raises(ValueError):
            index_with_sample_data.delete(
                namespace="test_namespace", delete_all=True, filter = {"key": "value"})

    def test_query_index(self, index_with_sample_data: IndexInterface) -> None:
        results = index_with_sample_data.query(id="id3", top_k=1, include_values=True)
        assert len(results) == 1
        assert results[0]["id"] == "id3"
        assert results[0]["values"] == pytest.approx([0.7, 0.8, 0.9], rel=1e-3)

    def test_query_dotproduct(self, index_dotproduct: IndexInterface) -> None:
        index_dotproduct.upsert([
            ("id1", [0.1, 0.2, 0.3]),
            ("id2", [1, 2, 3]),
            ("id3", [0.7, 0.8, 0.9]),
        ])
        query_vector = [0.1, 0.2, 0.3]
        results: List[MatchTypedDict] = index_dotproduct.query(
            vector=query_vector, top_k=2, include_values=True)
        assert len(results) == 2
        assert results[0]["id"] == "id2"
        assert results[1]["id"] == "id3"
        assert results[0]["score"] == pytest.approx(1.4, rel=1e-3)
        assert results[1]["score"] == pytest.approx(0.5, rel=1e-3)

    def test_query_cosine(self, index_cosine: IndexInterface) -> None:
        index_cosine.upsert([
            ("id1", [0.1, 0.2, 0.3]),
            ("id2", [1, 2, 3.3]),
            ("id3", [0.7, 0.8, 0.9]),
        ])
        query_vector = [0.1, 0.2, 0.3]
        results: List[MatchTypedDict] = index_cosine.query(
            vector=query_vector, top_k=2, include_values=True)
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[1]["id"] == "id2"

        assert results[0]["score"] == pytest.approx(1.0, rel=1e-3)
        assert results[1]["score"] == pytest.approx(0.998, rel=1e-3)

    def test_query_euclidean(self, index_euclidean: IndexInterface) -> None:
        index_euclidean.upsert([
            ("id1", [0.1, 0.2, 0.3]),
            ("id2", [1, 2, 3]),
            ("id3", [0.7, 0.8, 0.9]),
        ])
        query_vector = [0.1, 0.2, 0.3]
        results: List[MatchTypedDict] = index_euclidean.query(
            vector=query_vector, top_k=2, include_values=True)
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[1]["id"] == "id3"
        assert results[0]["score"] == pytest.approx(0.0, rel=1e-3)
        assert results[1]["score"] == pytest.approx(1.039, rel=1e-3)