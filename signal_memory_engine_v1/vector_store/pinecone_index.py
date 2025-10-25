# vector_store/pinecone_index.py

from pinecone import Pinecone, ServerlessSpec


def init_pinecone_index(
    api_key: str,
    environment: str,
    index_name: str,
    dimension: int,
    metric: str = "cosine",
    cloud: str = "aws",
):
    """
    Ensure a Pinecone index exists with the given name, dimension, and metric.
    If an existing index has the wrong dimension, it will be deleted and recreated.
    Returns a live Index client.
    """
    # 1) instantiate Pinecone client
    pc = Pinecone(api_key=api_key, environment=environment)

    # 2) list existing indexes
    existing = pc.list_indexes().names()

    # 3) if exists but wrong dim, delete
    if index_name in existing:
        desc = pc.describe_index(name=index_name)
        if desc.dimension != dimension:
            print(
                f"Index '{index_name}' exists with dimension={desc.dimension}, expected={dimension}. Recreating."
            )
            pc.delete_index(name=index_name)
            existing.remove(index_name)

    # 4) create if missing
    if index_name not in existing:
        print(f"Creating index '{index_name}' (dim={dimension}, metric={metric})")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=environment),
        )

    # 5) return the Index client
    return pc.Index(index_name)
