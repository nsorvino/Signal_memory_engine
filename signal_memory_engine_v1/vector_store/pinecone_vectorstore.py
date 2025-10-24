# vector_store/pinecone_vectorstore.py

from typing import List, Optional, Dict, Any
from pinecone import Pinecone
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever


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
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
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
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
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
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store."""
        # Get embeddings
        embeddings = self.embedding.embed_documents(texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            metadata[self.text_key] = text
            
            vectors.append({
                "id": f"doc_{i}",
                "values": embedding,
                "metadata": metadata,
            })
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
        
        return [f"doc_{i}" for i in range(len(texts))]
    
    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Return a retriever for this vector store."""
        return PineconeRetriever(vectorstore=self, **kwargs)


class PineconeRetriever(BaseRetriever):
    """A retriever for PineconeVectorStore."""
    
    def __init__(self, vectorstore: PineconeVectorStore, **kwargs: Any):
        super().__init__()
        self.vectorstore = vectorstore
        self.search_kwargs = kwargs.get("search_kwargs", {"k": 4})
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query."""
        k = self.search_kwargs.get("k", 4)
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query."""
        return self._get_relevant_documents(query)