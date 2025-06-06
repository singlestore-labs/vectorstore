from importlib import metadata
import pytest
from singlestoredb import connect
from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool

from vectorstore import DeletionProtection, Metric, VectorDB


class TestVectorDB:
    def test_initialize_with_connection(self, clean_db_connection: Connection):
        db: VectorDB = VectorDB(connection=clean_db_connection)
        assert len(db.list_indexes()) == 0
        curr = clean_db_connection.cursor()
        curr.execute("show tables")
        result = curr.fetchone()
        assert result is not None
        assert result[0] == "vector_indexes"
        curr.close()

    def test_initialize_with_connection_params(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        assert len(db.list_indexes()) == 0
        conn = connect(**clean_connection_params)
        curr = conn.cursor()
        curr.execute("show tables")
        result = curr.fetchone()
        assert result is not None
        assert result[0] == "vector_indexes"
        curr.close()

    def test_connection_attributes(
        self, docker_server_url: str, clean_connection_params: dict
    ):
        _: VectorDB = VectorDB(**clean_connection_params)
        conn = connect(docker_server_url)
        curr = conn.cursor()
        curr.execute(
            "SELECT CONNECTION_ID, ATTRIBUTE_NAME, ATTRIBUTE_VALUE "
            "from INFORMATION_SCHEMA.LMV_CONNECTION_ATTRIBUTES "
            "WHERE ATTRIBUTE_NAME in ('_connector_name', '_connector_version') "
            "ORDER BY CONNECTION_ID"
        )
        results = curr.fetchall()
        curr.close()
        conn.close()
        assert results is not None
        found_name = False
        found_version = False
        for result in results:
            if result[1] == "_connector_name" and result[2] == "vectorstore python sdk":
                found_name = True
            elif result[1] == "_connector_version" and result[2] == metadata.version(
                "singlestore-vectorstore"
            ):
                found_version = True
        assert found_name is True
        assert found_version is True

    def test_initialize_with_connection_pool(self, clean_connection_pool: Pool):
        conn = clean_connection_pool.connect()
        curr = conn.cursor()
        curr.execute("show tables")
        result = curr.fetchone()
        curr.close()
        conn.close()
        assert result is None
        db: VectorDB = VectorDB(connection_pool=clean_connection_pool)
        assert len(db.list_indexes()) == 0
        conn = clean_connection_pool.connect()
        curr = conn.cursor()
        curr.execute("show tables")
        result = curr.fetchone()
        curr.close()
        conn.close()
        assert result is not None
        assert result[0] == "vector_indexes"

    def test_create_index1(slef, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        index = db.create_index(
            name="test_index1",
            dimension=128,
            metric="cosine",
            deletion_protection="enabled",
        )
        assert index.name == "test_index1"
        assert index.dimension == 128
        assert index.metric == Metric.COSINE
        assert index.deletion_protection == DeletionProtection.ENABLED
        assert index.tags == {}
        assert index.use_vector_index is False
        assert index.vector_index_options == {"metric_type": "DOT_PRODUCT"}
        conn = connect(**clean_connection_params)
        curr = conn.cursor()
        curr.execute(
            "select name, dimension, metric, deletion_protection, use_vector_index, vector_index_options, tags from vector_indexes where name = 'test_index1'"
        )
        result = curr.fetchone()
        assert result is not None
        assert result[0] == "test_index1"
        assert result[1] == 128
        assert result[2] == Metric.COSINE.value
        assert result[3] == 1
        assert result[4] == 0
        assert result[5] == {"metric_type": "DOT_PRODUCT"}
        assert result[6] == {}
        curr.close()
        conn.close()

    def test_create_index2(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        index = db.create_index(
            name="test_index2",
            dimension=256,
            metric=Metric.EUCLIDEAN,
            deletion_protection=DeletionProtection.DISABLED,
            use_vector_index=True,
            vector_index_options={
                "index_type": "IVF_PQFS",
                "nlist": 1024,
                "nprobe": 20,
            },
            tags={"tag1": "value1", "tag2": "value2"},
        )
        assert index.name == "test_index2"
        assert index.dimension == 256
        assert index.metric == Metric.EUCLIDEAN
        assert index.deletion_protection == DeletionProtection.DISABLED
        assert index.tags == {"tag1": "value1", "tag2": "value2"}
        assert index.use_vector_index is True
        assert index.vector_index_options == {
            "index_type": "IVF_PQFS",
            "nlist": 1024,
            "nprobe": 20,
            "metric_type": "EUCLIDEAN_DISTANCE",
        }
        conn = connect(**clean_connection_params)
        curr = conn.cursor()
        curr.execute(
            "select name, dimension, metric, deletion_protection, use_vector_index, vector_index_options, tags from vector_indexes where name = 'test_index2'"
        )
        result = curr.fetchone()
        assert result is not None
        assert result[0] == "test_index2"
        assert result[1] == 256
        assert result[2] == Metric.EUCLIDEAN.value
        assert result[3] == 0
        assert result[4] == 1
        assert result[5] == {
            "index_type": "IVF_PQFS",
            "nlist": 1024,
            "nprobe": 20,
            "metric_type": "EUCLIDEAN_DISTANCE",
        }
        assert result[6] == {"tag1": "value1", "tag2": "value2"}
        curr.close()
        conn.close()

    def test_create_index3(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        index = db.create_index(
            name="test_index3",
            dimension=512,
            metric=Metric.COSINE,
            deletion_protection=DeletionProtection.DISABLED,
            use_vector_index=True,
            vector_index_options={
                "index_type": "IVF_FLAT",
                "nlist": 1024,
                "nprobe": 20,
            },
            tags={"tag1": "value1", "tag2": "value2"},
        )
        assert index.name == "test_index3"
        assert index.dimension == 512
        assert index.metric == Metric.COSINE
        assert index.deletion_protection == DeletionProtection.DISABLED
        assert index.tags == {"tag1": "value1", "tag2": "value2"}
        assert index.use_vector_index is True
        assert index.vector_index_options == {
            "index_type": "IVF_FLAT",
            "nlist": 1024,
            "nprobe": 20,
            "metric_type": "DOT_PRODUCT",
        }
        conn = connect(**clean_connection_params)
        curr = conn.cursor()
        curr.execute(
            "select name, dimension, metric, deletion_protection, use_vector_index, vector_index_options, tags from vector_indexes where name = 'test_index3'"
        )
        result = curr.fetchone()
        assert result is not None
        assert result[0] == "test_index3"
        assert result[1] == 512
        assert result[2] == Metric.COSINE.value
        assert result[3] == 0
        assert result[4] == 1
        assert result[5] == {
            "index_type": "IVF_FLAT",
            "nlist": 1024,
            "nprobe": 20,
            "metric_type": "DOT_PRODUCT",
        }
        assert result[6] == {"tag1": "value1", "tag2": "value2"}
        curr.close()
        conn.close()

    def test_delete_index_success(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        db.create_index(
            name="test_index4",
            dimension=128,
            metric=Metric.COSINE,
            deletion_protection=DeletionProtection.DISABLED,
        )
        connection = connect(**clean_connection_params)
        curr = connection.cursor()
        curr.execute("select name from vector_indexes where name = 'test_index4'")
        result = curr.fetchone()
        assert result is not None
        assert result[0] == "test_index4"
        curr.execute("show tables")
        result = curr.fetchall()
        assert len(result) == 2
        curr.close()
        connection.close()
        db.delete_index("test_index4")
        indexes = db.list_indexes()
        assert len(indexes) == 0
        connection = connect(**clean_connection_params)
        curr = connection.cursor()
        curr.execute("select name from vector_indexes where name = 'test_index4'")
        result = curr.fetchone()
        assert result is None
        curr.execute("show tables")
        result = curr.fetchall()
        assert len(result) == 1
        assert result[0][0] == "vector_indexes"
        curr.close()
        connection.close()

    def test_delete_index_protection(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        db.create_index(
            name="test_index5",
            dimension=128,
            metric=Metric.COSINE,
            deletion_protection=DeletionProtection.ENABLED,
        )
        with pytest.raises(Exception):
            db.delete_index("test_index5")
        connection = connect(**clean_connection_params)
        curr = connection.cursor()
        curr.execute("select name from vector_indexes where name = 'test_index5'")
        result = curr.fetchone()
        assert result is not None
        assert result[0] == "test_index5"
        curr.execute("show tables")
        result = curr.fetchall()
        assert len(result) == 2
        curr.close()
        connection.close()

    def test_has_index(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        db.create_index(
            name="test_index6",
            dimension=128,
            metric=Metric.COSINE,
            deletion_protection=DeletionProtection.DISABLED,
        )
        assert db.has_index("test_index6") is True
        assert db.has_index("test_index7") is False

    def test_list_indexes(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        db.create_index(
            name="test_index8",
            dimension=128,
            metric=Metric.COSINE,
            deletion_protection=DeletionProtection.DISABLED,
            tags={"tag1": "value1", "tag2": "value2"},
            use_vector_index=True,
            vector_index_options={
                "index_type": "IVF_FLAT",
                "nlist": 1024,
                "nprobe": 20,
            },
        )
        db.create_index(
            name="test_index9",
            dimension=256,
            metric=Metric.EUCLIDEAN,
            deletion_protection=DeletionProtection.ENABLED,
        )
        indexes = db.list_indexes()
        assert len(indexes) == 2
        assert indexes[0].name == "test_index8"
        assert indexes[1].name == "test_index9"
        assert indexes[0].dimension == 128
        assert indexes[1].dimension == 256
        assert indexes[0].metric == Metric.COSINE
        assert indexes[1].metric == Metric.EUCLIDEAN
        assert indexes[0].deletion_protection == DeletionProtection.DISABLED
        assert indexes[1].deletion_protection == DeletionProtection.ENABLED
        assert indexes[0].tags == {"tag1": "value1", "tag2": "value2"}
        assert indexes[1].tags == {}
        assert indexes[0].use_vector_index is True
        assert indexes[1].use_vector_index is False
        assert indexes[0].vector_index_options == {
            "index_type": "IVF_FLAT",
            "nlist": 1024,
            "nprobe": 20,
            "metric_type": "DOT_PRODUCT",
        }
        assert indexes[1].vector_index_options == {
            "metric_type": "EUCLIDEAN_DISTANCE",
        }

    def test_describe_index1(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        db.create_index(
            name="test_index10",
            dimension=128,
            metric=Metric.COSINE,
            deletion_protection=DeletionProtection.DISABLED,
            tags={"tag1": "value1", "tag2": "value2"},
            use_vector_index=True,
            vector_index_options={
                "index_type": "IVF_FLAT",
                "nlist": 1024,
                "nprobe": 20,
            },
        )
        index = db.describe_index("test_index10")
        assert index.name == "test_index10"
        assert index.dimension == 128
        assert index.metric == Metric.COSINE
        assert index.deletion_protection == DeletionProtection.DISABLED
        assert index.tags == {"tag1": "value1", "tag2": "value2"}
        assert index.use_vector_index is True
        assert index.vector_index_options == {
            "index_type": "IVF_FLAT",
            "nlist": 1024,
            "nprobe": 20,
            "metric_type": "DOT_PRODUCT",
        }

    def test_describe_index2(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        db.create_index(
            name="test_index11",
            dimension=256,
            metric=Metric.DOTPRODUCT,
            deletion_protection=DeletionProtection.ENABLED,
        )
        index = db.describe_index("test_index11")
        assert index.name == "test_index11"
        assert index.dimension == 256
        assert index.metric == Metric.DOTPRODUCT
        assert index.deletion_protection == DeletionProtection.ENABLED
        assert index.tags == {}
        assert index.use_vector_index is False
        assert index.vector_index_options == {
            "metric_type": "DOT_PRODUCT",
        }

    def test_describe_index_not_found(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        with pytest.raises(Exception):
            db.describe_index("test_index_not_found")

    def test_configure_index1(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        db.create_index(
            name="test_index12",
            dimension=128,
            metric=Metric.COSINE,
            deletion_protection=DeletionProtection.DISABLED,
            tags={"tag1": "value1", "tag2": "value2"},
            use_vector_index=True,
            vector_index_options={
                "index_type": "IVF_FLAT",
                "nlist": 1024,
                "nprobe": 20,
            },
        )
        index = db.configure_index(
            "test_index12",
            deletion_protection=DeletionProtection.ENABLED,
            tags={"tag3": "value3", "tag4": "value4"},
            use_vector_index=False,
        )
        assert index.name == "test_index12"
        assert index.dimension == 128
        assert index.metric == Metric.COSINE
        assert index.deletion_protection == DeletionProtection.ENABLED
        assert index.tags == {"tag3": "value3", "tag4": "value4"}
        assert index.use_vector_index is False
        assert index.vector_index_options == {
            "metric_type": "DOT_PRODUCT",
        }

    def test_configure_index2(self, clean_connection_params: dict):
        db: VectorDB = VectorDB(**clean_connection_params)
        db.create_index(
            name="test_index13",
            dimension=256,
            metric=Metric.EUCLIDEAN,
            deletion_protection=DeletionProtection.DISABLED,
            tags={"tag1": "value1", "tag2": "value2"},
            use_vector_index=True,
            vector_index_options={
                "index_type": "IVF_FLAT",
                "nlist": 1024,
                "nprobe": 20,
            },
        )
        index = db.configure_index("test_index13")
        assert index.name == "test_index13"
        assert index.dimension == 256
        assert index.metric == Metric.EUCLIDEAN
        assert index.deletion_protection == DeletionProtection.DISABLED
        assert index.tags == {"tag1": "value1", "tag2": "value2"}
        assert index.use_vector_index is True
        assert index.vector_index_options == {
            "metric_type": "EUCLIDEAN_DISTANCE",
            "index_type": "IVF_FLAT",
            "nlist": 1024,
            "nprobe": 20,
        }
