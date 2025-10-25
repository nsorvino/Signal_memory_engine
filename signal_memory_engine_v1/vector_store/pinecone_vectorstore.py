# vector_store/pinecone_vectorstore.py

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pinecone import Pinecone


class PineconeVectorStore(VectorStore):
    """A Pinecone vector store implementation that avoids langchain_pinecone dependency issues."""

    def __init__(
        self,
        index: Any,  # Pinecone Index object
        embedding: Embeddings,
        text_key: str = "content",
    ):
        self.index = index
        self.embedding = embedding
        self.text_key = text_key

    @classmethod
    def from_existing_index(
        cls,
        index_name: str,
        embedding: Embeddings,
        text_key: str = "content",
        api_key: str | None = None,
        environment: str | None = None,
    ) -> "PineconeVectorStore":
        """Create a PineconeVectorStore from an existing index."""
        # Use environment variables if not provided
        if not api_key:
            import os

            api_key = os.getenv("PINECONE_API_KEY")
        if not environment:
            import os

            environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

        if not api_key:
            raise ValueError("PINECONE_API_KEY must be provided or set as environment variable")

        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key, environment=environment)
        index = pc.Index(index_name)

        return cls(index=index, embedding=embedding, text_key=text_key)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Perform similarity search."""
        # Get query embedding
        query_embedding = self.embedding.embed_query(query)

        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            filter=filter,
            include_metadata=True,
        )

        # Convert to Document objects
        documents = []
        for match in results.matches:
            metadata = match.metadata or {}
            content = metadata.get(self.text_key, "")
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store."""
        # Get embeddings
        embeddings = self.embedding.embed_documents(texts)

        # Prepare vectors for Pinecone
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            metadata[self.text_key] = text

            vectors.append(
                {
                    "id": f"doc_{i}",
                    "values": embedding,
                    "metadata": metadata,
                }
            )

        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)

        return [f"doc_{i}" for i in range(len(texts))]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        text_key: str = "content",
        api_key: str | None = None,
        environment: str | None = None,
        index_name: str | None = None,
        **kwargs: Any,
    ) -> "PineconeVectorStore":
        """Create a PineconeVectorStore from a list of texts."""
        if not index_name:
            raise ValueError("index_name is required for from_texts")

        # Create the vector store
        vectorstore = cls.from_existing_index(
            index_name=index_name,
            embedding=embedding,
            text_key=text_key,
            api_key=api_key,
            environment=environment,
        )

        # Add the texts
        vectorstore.add_texts(texts, metadatas=metadatas, **kwargs)

        return vectorstore

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a retriever for this vector store."""
        return PineconeRetriever(vectorstore=self, **kwargs)


class PineconeRetriever(BaseRetriever):
    """A retriever for PineconeVectorStore."""

    def __init__(self, vectorstore: PineconeVectorStore, **kwargs: Any):
        super().__init__()
        self._vectorstore = vectorstore
        self._search_kwargs = kwargs.get("search_kwargs", {"k": 4})

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Get relevant documents for a query."""
        k = self._search_kwargs.get("k", 4)
        return self._vectorstore.similarity_search(query, k=k)
