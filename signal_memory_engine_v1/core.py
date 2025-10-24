import os

from dotenv import load_dotenv

# Quiet HF tokenizers fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from vector_store import PineconeVectorStore as LC_Pinecone

load_dotenv()


def build_qa_chain(
    pinecone_api_key: str,  # kept for signature parity (unused here)
    pinecone_env: str,  # kept for parity (unused)
    index_name: str,
    openai_api_key: str,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    k: int = 3,
    text_key: str = "content",
):
    # Basic validation
    if not index_name or not openai_api_key:
        raise RuntimeError("Missing index_name or OPENAI_API_KEY")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)

    # Vectorstore (community Pinecone vectorstore; no monkey-patch required)
    vectorstore = LC_Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        text_key=text_key,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # LLM (newer signature: model + api_key)
    llm = ChatOpenAI(
        model=llm_model,
        api_key=openai_api_key,
        temperature=0.7,
        max_tokens=256,
    )

    # Prompt + LCEL chain
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Use the given context to answer the question. "
                "If you don't know the answer, say you don't know. "
                "Use three sentences maximum and keep the answer concise. "
                "Context: {context}",
            ),
            ("human", "{input}"),
        ]
    )

    combine = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, combine)
    return qa_chain, vectorstore


if __name__ == "__main__":
    pinecone_api_key = os.getenv("PINECONE_API_KEY")  # unused here but kept for parity
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV", "us-west1-gcp")
    index_name = os.getenv("PINECONE_INDEX", "signal-engine")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    k = 3

    if pinecone_api_key is None or pinecone_env is None or openai_api_key is None:
        raise RuntimeError("Missing Pinecone/OpenAI configuration")

    qa, _ = build_qa_chain(
        pinecone_api_key=pinecone_api_key or "",
        pinecone_env=pinecone_env or "",
        index_name=index_name,
        openai_api_key=openai_api_key or "",
        k=k,
    )

    question = "What is emotional recursion?"
    result = qa.invoke({"input": question})

    # Print the answer (and optionally show which docs were used)
    answer = (
        result.get("answer") if isinstance(result, dict) and "answer" in result else str(result)
    )
    print(f"Q: {question}\nA: {answer}")

    # Optional: print brief context provenance (first 2 docs)
    ctx = result.get("context") if isinstance(result, dict) else None
    if ctx:
        print("\nContext sources:")
        for i, doc in enumerate(ctx[:2], start=1):
            meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source") or meta.get("id") or meta.get("path") or "unknown"
            print(f"  {i}. {src}")
