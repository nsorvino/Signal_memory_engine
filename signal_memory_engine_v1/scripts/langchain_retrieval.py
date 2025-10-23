#!/usr/bin/env python
import logging
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def flag_from_score(score: float) -> str:
    if score > 0.8:
        return "concern"
    elif score > 0.5:
        return "drifting"
    else:
        return "stable"

SUGGESTIONS = {
    "stable":   "No action needed.",
    "drifting": "Consider sending a check-in message.",
    "concern":  "Recommend escalation or a one-on-one conversation."
}

def build_qa_chain(
    pinecone_api_key: str,   # kept for parity; not used by LC vectorstore
    pinecone_env: str,       # kept for parity; not used here
    index_name: str,
    openai_api_key: str,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = "gpt-3.5-turbo",
    k: int = 3,
):
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = LC_Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=index_name,
        text_key="content",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(
        model=llm_model,
        api_key=openai_api_key,
        temperature=0.7,
        max_tokens=256,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Use the given context to answer the question. "
         "If you don't know the answer, say you don't know. "
         "Use three sentences maximum and keep the answer concise. "
         "Context: {context}"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)
    logger.debug("QA chain initialized without Pinecone monkey-patch")
    return qa_chain, vectorstore
