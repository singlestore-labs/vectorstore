# SingleStore VectorStore

A high-performance vector database library for storing and querying vector embeddings in SingleStore DB. Designed to efficiently manage and search through high-dimensional vector data for AI/ML applications, semantic search, and recommendation systems.

## Table of Contents

- [Installation](#installation)
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Connecting to SingleStore](#connecting-to-singlestore)
- [Creating and Managing Indexes](#creating-and-managing-indexes)
- [Working with Vectors](#working-with-vectors)
- [Querying Vectors](#querying-vectors)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)

## Installation

Install the package using pip:

```bash
pip install singlestore-vectorstore
```

## Overview

SingleStore VectorStore is a Python library that provides:

- Simple API for vector similarity search
- Efficient indexing for high-dimensional vectors
- Support for multiple distance metrics (Cosine, Dot Product, Euclidean)
- Metadata filtering capabilities
- Connection pooling for performance
- Namespace support for organizing vectors

## Getting Started

### Basic Usage

```python
from vectorstore import VectorDB, Metric, Vector

# Initialize the VectorDB
db = VectorDB(
    host="localhost",
    user="root",
    password="password",
    database="embeddings_db"
)

# Create an index
db.create_index(
    name="my_embeddings",
    dimension=1536,  # e.g., for OpenAI embeddings
    metric=Metric.COSINE,
)

# Get a reference to the index
index = db.Index("my_embeddings")

# Add vectors to the index
vectors = [
    Vector(id="doc1", vector=[0.1, 0.2, 0.3, ...], metadata={"source": "article"}),
    Vector(id="doc2", vector=[0.2, 0.3, 0.4, ...], metadata={"source": "webpage"})
]
index.upsert(vectors)

# Find similar vectors
results = index.query(
    vector=[0.15, 0.25, 0.35, ...],
    top_k=5,
    include_metadata=True
)

# Print results
for match in results:
    print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match['metadata']}")
```

## Connecting to SingleStore

### Connection Options

There are several ways to connect to SingleStore DB:

#### 1. Direct Connection Parameters

Direct connection parameters can be passed as separate parameters:
```python
from vectorstore import VectorDB

db = VectorDB(
    host="localhost",
    port=3306,
    user="root",
    password="password",
    database="vectors"
)
```

 Or as a connection URL:
```python
from vectorstore import VectorDB

db = VectorDB(
    host="root:password@localhost:3306/vectors"
)
```

Or as environment variables:
```python
os.environ['SingleStore_URL'] = 'me:p455w0rd@s2-host.com/my_db'
db = VectorDB()
```

The VectorDB supports all ways of connection supported by original [singlestordb](https://singlestoredb-python.labs.singlestore.com/api.html) python client.

#### 2. Existing Connection

```python
from singlestoredb import connect
from vectorstore import VectorDB

# Create a connection
connection = connect(
    host="localhost",
    user="root",
    password="password",
    database="vectors"
)

# Use the existing connection
db = VectorDB(connection=connection)
```

#### 3. Connection Pool (Recommended for Production)

```python
from sqlalchemy.pool import QueuePool
from singlestoredb import connect
from vectorstore import VectorDB

# Create a connection pool
def create_connection():
    return connect(
        host="localhost",
        user="root",
        password="password",
        database="vectors"
    )

connection_pool = QueuePool(
    creator=create_connection,
    pool_size=10,
    max_overflow=20,
    timeout=30
)

# Use the connection pool
db = VectorDB(connection_pool=connection_pool)
```

## Creating and Managing Indexes

### Creating an Index

```python
from vectorstore import VectorDB, Metric, DeletionProtection

db = VectorDB(host="localhost", user="root", password="password", database="vectors")

# Create a simple index
basic_index = db.create_index(
    name="basic_index",
    dimension=1536,
)

# Create a more customized index
custom_index = db.create_index(
    name="custom_index",
    dimension=768,
    metric=Metric.EUCLIDEAN,
    deletion_protection=DeletionProtection.ENABLED,
    tags={"model": "sentence-transformers", "version": "v1.0"},
    use_vector_index=True,
    vector_index_options={
        "index_type": "IVF_PQFS",
        "nlist": 1024,
        "nprobe": 20
    }
)
```

### Vector Index Options

When creating an index with `use_vector_index=True`, you can configure various index types and parameters to optimize for your specific use case. SingleStore supports several vector index types, each with different performance characteristics:

```python
vector_index_options={
    "index_type": "IVF_FLAT",  # Specify the index type
    "nlist": 1024,             # Number of clusters/centroids
    "nprobe": 20,              # Number of clusters to search during query time
    # Additional parameters specific to each index type...
}
```

#### Supported Index Types

1. **FLAT**
   - Brute force approach that compares against every vector
   - Highest accuracy but slowest for large datasets
   - No additional parameters required
   - Best for: Small datasets or when accuracy is critical

2. **IVF_FLAT** (Inverted File with Flat Quantizer)
   - Uses clustering to accelerate searches
   - Good balance of quality and performance
   - Parameters:
     - `nlist`: Number of centroids/clusters (default 100, higher values improve accuracy but slow down indexing)
     - `nprobe`: Number of clusters to search at query time (default 1, higher values improve accuracy but slow down search)
   - Best for: Medium-sized datasets with moderate query performance requirements

3. **IVF_SQ** (Inverted File with Scalar Quantization)
   - Compresses vectors to reduce memory usage
   - Parameters:
     - `nlist`, `nprobe`: Same as IVF_FLAT
     - `qtype`: Quantizer type, either "QT8" (8-bit) or "QT4" (4-bit)
   - Best for: Large datasets where memory usage is a concern

4. **IVF_PQ** (Inverted File with Product Quantization)
   - Advanced compression technique that divides vectors into subvectors
   - Parameters:
     - `nlist`, `nprobe`: Same as IVF_FLAT
     - `m`: Number of subvectors (default: dimension / 2)
     - `nbits`: Bits per subvector (default: 8)
   - Best for: Very large datasets where memory usage is critical

5. **IVF_PQFS** (Inverted File with PQ Fast Scan)
   - Optimized version of IVF_PQ with SIMD acceleration
   - Parameters:
     - `nlist`, `nprobe`: Same as IVF_FLAT
     - `m`: Number of subvectors (must be multiple of 4)
     - `nbits`: Bits per subvector (must be 8)
   - Best for: Production systems with large datasets and high query throughput

6. **HNSW** (Hierarchical Navigable Small World)
   - Graph-based approach that builds navigation network between vectors
   - Very fast queries but slower index building
   - Parameters:
     - `M`: Number of edges per node (default: 12)
     - `efConstruction`: Size of dynamic list during construction (default: 40)
     - `ef`: Size of dynamic list during search (default: 10)
     - `random_seed`: Random seed for reproducibility (default: current time)
   - Best for: Applications requiring extremely fast search on moderate-sized datasets

#### Parameter Tuning Guidelines

- **Increasing `nlist`**: Improves search speed but requires more memory and longer index build time
- **Increasing `nprobe`**: Improves accuracy but slows down searches
- **For IVF_PQ/PQFS**:
  - Lower `m` values: Faster search but lower accuracy
  - Higher `m` values: Better accuracy but slower search
- **For HNSW**:
  - Higher `M` values: Better accuracy but larger index size and longer build time
  - Higher `ef` values: Better accuracy but slower search

For complete details on vector indexing options, see the [SingleStore Vector Indexing documentation](https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/).

### Listing Indexes

```python
# Get all indexes
indexes = db.list_indexes()

# Print index details
for idx in indexes:
    print(f"Index: {idx.name}, Dimension: {idx.dimension}, Metric: {idx.metric}")
```

### Describing an Index

```python
# Get detailed information about an index
index_info = db.describe_index("my_index")
print(f"Name: {index_info.name}")
print(f"Dimension: {index_info.dimension}")
print(f"Metric: {index_info.metric}")
print(f"Deletion Protection: {index_info.deletion_protection}")
print(f"Tags: {index_info.tags}")
print(f"Uses Vector Index: {index_info.use_vector_index}")
print(f"Vector Index Options: {index_info.vector_index_options}")
```

### Configuring an Index

```python
# Update index settings
db.configure_index(
    name="my_index",
    deletion_protection=DeletionProtection.ENABLED,
    tags={"updated": "true", "version": "v2.0"},
    use_vector_index=True,
    vector_index_options={
        "index_type": "IVF_FLAT",
        "nlist": 2048
    }
)
```

### Checking If an Index Exists

```python
if db.has_index("my_index"):
    print("Index exists")
else:
    print("Index doesn't exist")
```

### Deleting an Index

```python
# Delete an index
db.delete_index("my_index")

# This will fail if deletion protection is enabled
try:
    db.delete_index("protected_index")
except ValueError as e:
    print(f"Could not delete: {e}")
```

## Working with Vectors

### Different Ways to Represent Vectors

```python
from vectorstore import Vector

# Method 1: Using Vector class
vectors = [
    Vector(id="vec1", vector=[0.1, 0.2, 0.3], metadata={"category": "A"}),
    Vector(id="vec2", vector=[0.4, 0.5, 0.6], metadata={"category": "B"})
]

# Method 2: Using tuples (id, values)
vectors_tuples = [
    ("vec3", [0.7, 0.8, 0.9]),
    ("vec4", [0.10, 0.11, 0.12])
]

# Method 3: Using tuples with metadata (id, values, metadata)
vectors_with_meta = [
    ("vec5", [0.13, 0.14, 0.15], {"category": "C"}),
    ("vec6", [0.16, 0.17, 0.18], {"category": "D"})
]

# Method 4: Using dictionaries
vectors_dict = [
    {"id": "vec7", "values": [0.19, 0.20, 0.21], "metadata": {"category": "E"}},
    {"id": "vec8", "values": [0.22, 0.23, 0.24], "metadata": {"category": "F"}}
]
```

### Inserting Vectors

```python
# Get index reference
index = db.Index("my_index")

# Insert vectors
count = index.upsert(vectors)
print(f"Inserted {count} vectors")

# Insert with namespace
index.upsert(vectors_tuples, namespace="group1")
index.upsert(vectors_with_meta, namespace="group2")
```

### Using Pandas DataFrames

```python
import pandas as pd

# Create a DataFrame with vector data
df = pd.DataFrame([
    {"id": "vec1", "values": [0.1, 0.2, 0.3], "metadata": {"category": "A"}},
    {"id": "vec2", "values": [0.4, 0.5, 0.6], "metadata": {"category": "B"}}
])

# Upsert from DataFrame
count = index.upsert_from_dataframe(df, namespace="pandas_import")
print(f"Imported {count} vectors from DataFrame")
```

### Updating Vectors

```python
# Update vector values
index.update(
    id="vec1",
    values=[0.25, 0.35, 0.45]
)

# Update metadata only
index.update(
    id="vec2",
    set_metadata={"category": "updated", "version": 2}
)

# Update both values and metadata with namespace
index.update(
    id="vec3",
    values=[0.55, 0.65, 0.75],
    set_metadata={"processed": True},
    namespace="group1"
)
```

### Fetching Vectors

```python
# Get vectors by ID
vectors = index.fetch(
    ids=["vec1", "vec2", "vec3"]
)

# Get vectors by ID with namespace
vectors_in_namespace = index.fetch(
    ids=["vec3", "vec4"],
    namespace="group1"
)

# Access vector data
for vec_id, vec_obj in vectors.items():
    print(f"ID: {vec_id}")
    print(f"Vector: {vec_obj.vector[:5]}...")  # Print first 5 elements
    print(f"Metadata: {vec_obj.metadata}")
```

### Deleting Vectors

```python
# Delete vectors by ID
index.delete(ids=["vec1", "vec2"])

# Delete vectors by ID in a namespace
index.delete(ids=["vec3", "vec4"], namespace="group1")

# Delete all vectors in a namespace
index.delete(delete_all=True, namespace="group2")

# Delete vectors matching a filter
index.delete(
    filter={"category": "A"},
    namespace="pandas_import"
)
```

### Listing Vector IDs

```python
# List all vector IDs
ids = index.list()

# List vectors with a prefix
ids_with_prefix = index.list(prefix="doc_")

# List vectors in a namespace
ids_in_namespace = index.list(namespace="group1")
```

### Getting Index Statistics

```python
# Get statistics about the index
stats = index.describe_index_stats()

print(f"Dimension: {stats['dimension']}")
print(f"Total Vector Count: {stats['total_vector_count']}")

# Namespace statistics
for ns_name, ns_stats in stats['namespaces'].items():
    print(f"Namespace: {ns_name}, Vectors: {ns_stats['vector_count']}")

# Get filtered statistics
filtered_stats = index.describe_index_stats(
    filter={"category": "A"}
)
```

## Querying Vectors

### Basic Query

```python
# Query by vector values
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=5
)

# Print results
for match in results:
    print(f"ID: {match['id']}, Score: {match['score']}")
```

### Query Options

```python
# Query with metadata and vector values in response
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    include_metadata=True,
    include_values=True
)

# Query by existing vector ID
results = index.query(
    id="vec1",  # Use this vector's values for the query
    top_k=5
)

# Query within a namespace
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    namespace="group1",
    top_k=5
)

# Query across multiple namespaces
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    namespaces=["group1", "group2"],
    top_k=5
)
```

### Query with Filtering

```python
# Simple equality filter
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    filter={"category": "A"}
)

# Comparison operators
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    filter={"year": {"$gt": 2020}}
)

# Multiple conditions with AND
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    filter={
        "$and": [
            {"category": "article"},
            {"year": {"$gte": 2020}}
        ]
    }
)

# Multiple conditions with OR
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    filter={
        "$or": [
            {"category": "article"},
            {"category": "blog"}
        ]
    }
)

# Check if field exists
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    filter={"author": {"$exists": True}}
)

# Collection operators
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    filter={"category": {"$in": ["article", "blog", "news"]}}
)
```

### Vector Index Control

Vector indexes significantly accelerate similarity searches, especially with large datasets, but there's always a tradeoff between search speed and accuracy. Higher accuracy settings typically result in slower searches, while faster searches may return slightly less optimal results.

```python
# Disable vector index for this query
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    disable_vector_index_use=True  # Force brute-force search for maximum accuracy
)

# Customize search options based on index type
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    search_options={
        # Parameters vary by index type
        "nprobe": 50,  # For IVF-based indexes
        "ef": 100      # For HNSW indexes
    }
)
```

#### Search Parameters by Index Type

Each vector index type supports different search-time parameters that control the speed vs. accuracy tradeoff:

**ALL TYPES**
    ```python
    search_options={
        "k": 50 # number of rows outputted by vector index scan. k must be >= top_k
    }
    ```

1. **FLAT**
   - No tunable search parameters (always performs exhaustive search)
   - Always returns exact results with highest accuracy
   
2. **IVF_FLAT, IVF_SQ, IVF_PQ, IVF_PQFS**
   ```python
   search_options={
       "nprobe": 20  # Number of clusters to search (higher = more accurate, but slower)
                     # Default is 1, common range: 5-100 depending on dataset size
   }
   ```

3. **HNSW**
   ```python
   search_options={
       "ef": 40      # Size of dynamic candidate list (higher = more accurate, but slower)
                     # Default is 10, common range: 20-200 depending on dataset size
   }
   ```

#### Tuning Tips

- Start with default values and increase gradually until you find the right balance
- For high recall requirements, use higher parameter values (higher `nprobe` or `ef`)
- For time-sensitive applications, use lower values
- Performance measurement example:
  ```python
  import time

  # Measure search time vs. accuracy tradeoff
  for nprobe in [1, 10, 50, 100]:
      start = time.time()
      results = index.query(
          vector=query_vector,
          top_k=10,
          search_options={"nprobe": nprobe}
      )
      end = time.time()
      print(f"nprobe={nprobe}, time={end-start:.4f}s")
      # Compare results with ground truth if available
  ```

For more details on vector index parameters, refer to the [SingleStore Vector Indexing documentation](https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/).

## Advanced Features

### Working with Different Distance Metrics

```python
# Create indexes with different metrics
cosine_index = db.create_index(
    name="cosine_index",
    dimension=1536,
    metric=Metric.COSINE  # Normalized dot product, best for comparing directions
)

dotproduct_index = db.create_index(
    name="dotproduct_index",
    dimension=1536,
    metric=Metric.DOTPRODUCT  # Raw dot product, good for comparing direction and magnitude
)

euclidean_index = db.create_index(
    name="euclidean_index",
    dimension=1536,
    metric=Metric.EUCLIDEAN  # Euclidean distance, good for spatial data
)
```

### Filter Types

```python
from vectorstore import (
    FilterTypedDict,  # Base filter type
    AndFilter,        # $and logical operator
    OrFilter,         # $or logical operator
    SimpleFilter,     # Direct field matching
    ExactMatchFilter, # Exact field value matching
    EqFilter,         # $eq comparison
    NeFilter,         # $ne comparison
    GtFilter,         # $gt comparison
    GteFilter,        # $gte comparison
    LtFilter,         # $lt comparison
    LteFilter,        # $lte comparison
    InFilter,         # $in collection operator
    NinFilter         # $nin collection operator
)

# Complex filter example
complex_filter: FilterTypedDict = {
    "$and": [
        {
            "$or": [
                {"category": "article"},
                {"category": "blog"}
            ]
        },
        {"year": {"$gte": 2020}},
        {"author": {"$exists": True}}
    ]
}

results = index.query(
    vector=[0.1, 0.2, 0.3, ...],
    top_k=10,
    filter=complex_filter
)
```

## API Reference

### Main Classes

- `VectorDB`: Main entry point for creating and managing vector indexes
- `IndexInterface`: Interface for interacting with a specific index
- `Vector`: Class representing a vector with ID, values, and metadata
- `IndexModel`: Configuration for an index

### Enums

- `Metric`: Similarity metrics (COSINE, DOTPRODUCT, EUCLIDEAN)
- `DeletionProtection`: Protection against accidental deletion (ENABLED, DISABLED)

## Best Practices

1. **Connection Management**:
   - Use connection pooling for production applications
   - Close connections properly when not using a pool

2. **Vector Indexing**:
   - Enable vector indexes for large datasets (use_vector_index=True)
   - Tune vector_index_options based on dataset size and query patterns

3. **Namespaces**:
   - Use namespaces to organize vectors by source, type, or domain
   - Query across multiple namespaces when relevant

4. **Batch Operations**:
   - Use batch operations for inserting multiple vectors
   - For large datasets, use upsert_from_dataframe with appropriate batch_size

5. **Metrics Selection**:
   - Cosine similarity is best for direction comparison (most common)
   - Dot product works well when magnitude matters
   - Euclidean distance is good for spatial data

6. **Deletion Protection**:
   - Enable deletion protection for production indexes
   - Configure indexes properly before adding large amounts of data

### Metadata Filtering

VectorStore supports powerful metadata filtering capabilities that let you narrow down vector searches based on their associated metadata.

#### Filter Types

1. **Simple Equality Filter**
   ```python
   # Find vectors where category is exactly "article"
   filter = {"category": "article"}
   ```

2. **Comparison Operators**
   ```python
   # Equal to
   filter = {"year": {"$eq": 2023}}

   # Not equal to
   filter = {"year": {"$ne": 2023}}

   # Greater than
   filter = {"year": {"$gt": 2020}}

   # Greater than or equal to
   filter = {"year": {"$gte": 2020}}

   # Less than
   filter = {"year": {"$lt": 2023}}

   # Less than or equal to
   filter = {"year": {"$lte": 2023}}
   ```

3. **Collection Operators**
   ```python
   # Value is in a specified array
   filter = {"category": {"$in": ["article", "blog", "news"]}}

   # Value is not in a specified array
   filter = {"category": {"$nin": ["video", "podcast"]}}
   ```

4. **Existence Checks**
   ```python
   # Field exists
   filter = {"author": {"$exists": True}}

   # Field does not exist
   filter = {"author": {"$exists": False}}
   ```

5. **Logical Operators**
   ```python
   # AND - all conditions must match
   filter = {
       "$and": [
           {"category": "article"},
           {"year": {"$gte": 2020}}
       ]
   }

   # OR - at least one condition must match
   filter = {
       "$or": [
           {"category": "article"},
           {"category": "blog"}
       ]
   }
   ```

6. **Combined Complex Filters**
   ```python
   # Articles or blogs from 2020 or later that have an author field
   filter = {
       "$and": [
           {
               "$or": [
                   {"category": "article"},
                   {"category": "blog"}
               ]
           },
           {"year": {"$gte": 2020}},
           {"author": {"$exists": True}}
       ]
   }
   ```

#### How Filtering Works

Metadata filters are translated into SQL expressions that filter results based on the JSON metadata stored with each vector. The filters are applied before distance calculation for SQL-level filtering, improving query efficiency.

#### Filter Usage

Filters can be used in multiple operations:

1. **In queries**:
   ```python
   results = index.query(
       vector=[0.1, 0.2, 0.3, ...],
       top_k=10,
       filter={"$and": [{"category": "article"}, {"year": {"$gte": 2020}}]}
   )
   ```

2. **For deletion operations**:
   ```python
   # Remove outdated vectors
   index.delete(
       filter={"status": "outdated"}
   )
   ```

3. **For statistical analysis**:
   ```python
   # Get statistics for a specific category
   stats = index.describe_index_stats(
       filter={"category": "article"}
   )
   ```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Roadmap

Future development plans include:

- Adding index-for-model support with hybrid search capabilities (combining text and vector embedding searches)