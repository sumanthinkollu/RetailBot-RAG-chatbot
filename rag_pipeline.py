"""
rag_pipeline.py — Sprint 2
Core RAG logic: load vector store, retrieve relevant product context, generate LLM answer.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

VECTORSTORE_PATH = "vectorstore/"

# ── Prompt Engineering ──────────────────────────────────────────────────────
SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful and friendly retail product assistant for a technology store.
Your job is to answer customer questions ONLY using the product information provided below.

Rules:
- Answer ONLY from the context provided. Do NOT use outside knowledge.
- If the answer is not found in the context, say: "I'm sorry, I don't have information about that. Please contact our support team."
- Be concise, friendly, and conversational.
- When comparing products, format clearly with key differences highlighted.
- Always mention price and availability if relevant.
- If asked about stock, mention current stock status.

--- PRODUCT CONTEXT ---
{context}
-----------------------

Customer Question: {question}

Your Answer:"""
)


def load_rag_chain(llm_type: str = "ollama", model_name: str = "llama3"):
    """
    Load the full RAG chain.
    
    Args:
        llm_type: "ollama" for local LLM, "openai" for OpenAI API
        model_name: model name to use
    
    Returns:
        RetrievalQA chain
    """
    # Load embeddings & vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 4},
    )

    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            "Vector store not found! Please run 'python ingest.py' first."
        )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Retriever: fetch top 3 most relevant product chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # LLM selection
    llm = _load_llm(llm_type, model_name)

    # Build RAG chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": SYSTEM_PROMPT},
    )

    return chain


def _load_llm(llm_type: str, model_name: str):
    """Load the specified LLM."""
    if llm_type == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.2)

    elif llm_type == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(model=model_name, temperature=0.2)

    elif llm_type == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.2)

    else:
        raise ValueError(f"Unknown llm_type: {llm_type}. Use 'ollama', 'openai', or 'groq'.")


def ask(chain, query: str) -> dict:
    """
    Ask a question and get an answer with source documents.
    
    Returns:
        dict with 'answer' and 'sources' keys
    """
    try:
        result = chain.invoke({"query": query})
        answer = result.get("result", "Sorry, I could not generate a response.")
        sources = result.get("source_documents", [])
        source_names = list({doc.metadata.get("name", "Unknown") for doc in sources})
        return {"answer": answer, "sources": source_names}
    except Exception as e:
        return {
            "answer": f"⚠️ Error generating response: {str(e)}",
            "sources": [],
        }
