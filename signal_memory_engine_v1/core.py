# core.py
import os
from typing import cast

import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def build_qa_chain(
    pinecone_api_key: str,
    pinecone_env: str,
    index_name: str,
    openai_api_key: str,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    k: int = 3,
    text_key: str = "content",
) -> tuple[object, LC_Pinecone]:
    """
    Build and return a retrieval chain + Pinecone vectorstore client.

    Returns:
        qa_chain: LangChain retrieval chain (created with create_retrieval_chain)
        vectorstore: Pinecone vectorstore (for similarity_search_with_score)
    """
    # 1) Validate env vars
    if not pinecone_api_key or not index_name or not openai_api_key:
        raise RuntimeError("Missing one of PINECONE_API_KEY, index_name, or OPENAI_API_KEY")

    # 2) Monkey-patch Pinecone
    if not hasattr(pinecone, "__version__"):
        pinecone.__version__ = "3.0.0"
    from pinecone.db_data.index import Index as PineconeIndexClass

    pinecone.Index = PineconeIndexClass

    # 3) Embeddings & Vectorstore
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = LC_Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=index_name,
        text_key=text_key,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 4) LLM
    llm = ChatOpenAI(
        model_name=llm_model,
        openai_api_key=openai_api_key,
        temperature=0.7,
        max_tokens=256,
    )

    # 5) Build retrieval chain using create_retrieval_chain
    # Define the system prompt
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
    )

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the document combination chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    return qa_chain, vectorstore


if __name__ == "__main__":
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV", "us-west1-gcp")
    index_name = os.getenv("PINECONE_INDEX", "signal-engine")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    k = 3

    if pinecone_api_key is None or pinecone_env is None or openai_api_key is None:
        raise RuntimeError("Missing Pinecone/OpenAI configuration")

    qa, _ = build_qa_chain(
        pinecone_api_key=cast(str, pinecone_api_key),
        pinecone_env=cast(str, pinecone_env),
        index_name=index_name,
        openai_api_key=cast(str, pinecone_env),
        k=k,
    )

    question = "What is emotional recursion?"
    out = qa.invoke({"input": question})
    answer = out.get("answer") if isinstance(out, dict) else str(out)

    print(f"Q: {question}\nA: {answer}")
