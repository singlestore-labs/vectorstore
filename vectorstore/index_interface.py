from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from vectorstore.stats import IndexStatsTypedDict

from .vector import (
    Vector,
    VectorMetadataTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
    VectorDictMetadataValue,
    VectorTypedDict
)

from .filter import FilterTypedDict

class IndexInterface(ABC):
    @abstractmethod
    def upsert(
        self,
        vectors: Union[
            List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]
        ],
        namespace: Optional[str] = None,
        **kwargs,
    ) -> int:
        pass

    @abstractmethod
    def upsert_from_dataframe(
        self, df, namespace: Optional[str] = None, batch_size: int = 500
    ):
        """Upserts a dataframe into the index.

        Args:
            df: A pandas dataframe with the following columns: id, values, sparse_values, and metadata.
            namespace: The namespace to upsert into.
            batch_size: The number of rows to upsert in a single batch.
            show_progress: Whether to show a progress bar.
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
        Args:
            ids (List[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                                Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                            If not specified, the default namespace is used.
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]


        The Delete operation deletes vectors from the index, from a single namespace.

        No error is raised if the vector id does not exist.

        Note: For any delete call, if namespace is not specified, the default namespace `""` is used.
        Since the delete operation does not error when ids are not present, this means you may not receive
        an error if you delete from the wrong namespace.

        Delete can occur in the following mutual exclusive ways:
        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
            (note that for this option delete all must be set to False)

        API reference: https://docs.pinecone.io/reference/delete_post

        Examples:
            >>> index.delete(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.delete(delete_all=True, namespace='my_namespace')
            >>> index.delete(filter={'key': 'value'}, namespace='my_namespace')


          Returns: An empty dictionary if the delete operation was successful.
        """
        pass

    @abstractmethod
    def fetch(self, ids: List[str], namespace: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        API reference: https://docs.pinecone.io/reference/fetch

        Examples:
            >>> index.fetch(ids=['id1', 'id2'], namespace='my_namespace')
            >>> index.fetch(ids=['id1', 'id2'])

        Args:
            ids (List[str]): The vector IDs to fetch.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]

        Returns: FetchResponse object which contains the list of Vector objects, and namespace name.
        """
        pass

    @abstractmethod
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
        """
        The Query operation searches a namespace, using a query vector.
        It retrieves the ids of the most similar items in a namespace, along with their similarity scores.

        API reference: https://docs.pinecone.io/reference/query

        Examples:
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> index.query(id='id1', top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace', filter={'key': 'value'})
            >>> index.query(id='id1', top_k=10, namespace='my_namespace', include_metadata=True, include_values=True)
            >>> index.query(vector=[1, 2, 3], sparse_vector={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>             top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], sparse_vector=SparseValues([1, 2], [0.2, 0.4]),
            >>>             top_k=10, namespace='my_namespace')

        Args:
            vector (List[float]): The query vector. This should be the same length as the dimension of the index
                                  being queried. Each `query()` request can contain only one of the parameters
                                  `id` or `vector`.. [optional]
            id (str): The unique ID of the vector to be used as a query vector.
                      Each `query()` request can contain only one of the parameters
                      `vector` or  `id`. [optional]
            top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str): The namespace to query vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (Dict[str, Union[str, float, int, bool, List, dict]):
                    The filter to apply. You can use vector metadata to limit your search.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
            include_values (bool): Indicates whether vector values are included in the response.
                                   If omitted the server will use the default value of False [optional]
            include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
                                     If omitted the server will use the default value of False  [optional]
            sparse_vector: (Union[SparseValues, Dict[str, Union[List[float], List[int]]]]): sparse values of the query vector.
                            Expected to be either a SparseValues object or a dict of the form:
                             {'indices': List[int], 'values': List[float]}, where the lists each have the same length.

        Returns: QueryResponse object which contains the list of the closest vectors as ScoredVector objects,
                 and namespace name.
        """
        pass

    @abstractmethod
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
        """The query_namespaces() method is used to make a query to multiple namespaces in parallel and combine the results into one result set.

        Since several asynchronous calls are made on your behalf when calling this method, you will need to tune the pool_threads and connection_pool_maxsize parameter of the Index constructor to suite your workload.

        Examples:

        ```python
        from pinecone import Pinecone

        pc = Pinecone(api_key="your-api-key")
        index = pc.Index(
            host="index-name",
            pool_threads=32,
            connection_pool_maxsize=32
        )

        query_vec = [0.1, 0.2, 0.3] # An embedding that matches the index dimension
        combined_results = index.query_namespaces(
            vector=query_vec,
            namespaces=['ns1', 'ns2', 'ns3', 'ns4'],
            metric="cosine",
            top_k=10,
            filter={'genre': {"$eq": "drama"}},
            include_values=True,
            include_metadata=True
        )
        for vec in combined_results.matches:
            print(vec.id, vec.score)
        print(combined_results.usage)
        ```

        Args:
            vector (List[float]): The query vector, must be the same length as the dimension of the index being queried.
            namespaces (List[str]): The list of namespaces to query.
            top_k (Optional[int], optional): The number of results you would like to request from each namespace. Defaults to 10.
            metric (str): Must be one of 'cosine', 'euclidean', 'dotproduct'. This is needed in order to merge results across namespaces, since the interpretation of score depends on the index metric type.
            filter (Optional[Dict[str, Union[str, float, int, bool, List, dict]]], optional): Pass an optional filter to filter results based on metadata. Defaults to None.
            include_values (Optional[bool], optional): Boolean field indicating whether vector values should be included with results. Defaults to None.
            include_metadata (Optional[bool], optional): Boolean field indicating whether vector metadata should be included with results. Defaults to None.
            sparse_vector (Optional[ Union[SparseValues, Dict[str, Union[List[float], List[int]]]] ], optional): If you are working with a dotproduct index, you can pass a sparse vector as part of your hybrid search. Defaults to None.

        Returns:
            QueryNamespacesResults: A QueryNamespacesResults object containing the combined results from all namespaces, as well as the combined usage cost in read units.
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
        The Update operation updates vector in a namespace.
        If a value is included, it will overwrite the previous value.
        If a set_metadata is included,
        the values of the fields specified in it will be added or overwrite the previous value.

        API reference: https://docs.pinecone.io/reference/update

        Examples:
            >>> index.update(id='id1', values=[1, 2, 3], namespace='my_namespace')
            >>> index.update(id='id1', set_metadata={'key': 'value'}, namespace='my_namespace')
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>              namespace='my_namespace')
            >>> index.update(id='id1', values=[1, 2, 3], sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4]),
            >>>              namespace='my_namespace')

        Args:
            id (str): Vector's unique id.
            values (List[float]): vector values to set. [optional]
            set_metadata (Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]]):
                metadata to set for vector. [optional]
            namespace (str): Namespace name where to update the vector.. [optional]
            sparse_values: (Dict[str, Union[List[float], List[int]]]): sparse values to update for the vector.
                           Expected to be either a SparseValues object or a dict of the form:
                           {'indices': List[int], 'values': List[float]} where the lists each have the same length.


        Returns: An empty dictionary if the update was successful.
        """
        pass

    @abstractmethod
    def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> IndexStatsTypedDict:
        """
        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        API reference: https://docs.pinecone.io/reference/describe_index_stats_post

        Examples:
            >>> index.describe_index_stats()
            >>> index.describe_index_stats(filter={'key': 'value'})

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See https://www.pinecone.io/docs/metadata-filtering/.. [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.
        """
        pass


    @abstractmethod
    def list(self, prefix: Optional[str], namespace: Optional[str]) -> List[str]:
        """
        The list operation accepts all of the same arguments as list_paginated, and returns a generator that yields
        a list of the matching vector ids in each page of results. It automatically handles pagination tokens on your
        behalf.

        Examples:
            >>> for ids in index.list(prefix='99', limit=5, namespace='my_namespace'):
            >>>     print(ids)
            ['99', '990', '991', '992', '993']
            ['994', '995', '996', '997', '998']
            ['999']

        Args:
            prefix (Optional[str]): The id prefix to match. If unspecified, an empty string prefix will
                                    be used with the effect of listing all ids in a namespace [optional]
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
            namespace (Optional[str]): The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]
        """
        pass

    @abstractmethod
    def fetch(self, ids: List[str], namespace: Optional[str] = None, **kwargs) -> Dict[str, Vector]:
        pass
