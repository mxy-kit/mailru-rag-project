import os
import pickle
import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ChatPromptTemplate: support both new and old import paths
try:
    from langchain_core.prompts import ChatPromptTemplate  # new path
except ImportError:
    from langchain.prompts import ChatPromptTemplate       # legacy fallback

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Try to import ChatGroq; it may exist in CI but we still want a safe fallback
try:
    from langchain_groq import ChatGroq  # real integration
except Exception:
    ChatGroq = None  # if import fails we will use DummyLLM

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_data_and_db(data_path, db_path, embedding_model):
    """Load serialized docs and FAISS index."""
    logging.info("Loading data and FAISS index...")
    with open(data_path, "rb") as f:
        docs = pickle.load(f)

   
    if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], dict) and "page_content" in docs[0]:
        docs = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in docs]

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    logging.info("Data and index successfully loaded.")
    return docs, db, embeddings


def format_docs(docs):
    """Join retrieved documents into a single context string."""
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)


class _DummyLLM:
    """Tiny LLM stub so tests can run without API keys/network.

    Must be callable so LC pipeline can coerce it via RunnableLambda.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, prompt):
        # Return a deterministic, harmless string for tests
        return "Stub LLM response. See https://help.mail.ru/"

    def invoke(self, prompt):
        # Keep invoke for direct calls; delegate to __call__
        return self.__call__(prompt)


def _has_groq_key() -> bool:
    """Check presence of GROQ_API_KEY in environment."""
    return bool(os.getenv("GROQ_API_KEY"))


def build_retriever(db=None, top_k=6, **kwargs):
    """
    Build a retriever from FAISS.
    Accept extra kwargs (docs, embeddings, etc.) to be compatible with tests.
    If db is None (e.g., in CI), return an empty retriever that safely composes in LC pipeline.
    """
    if db is not None:
        return db.as_retriever(search_kwargs={"k": top_k})

    class _EmptyRetriever:
        def __init__(self, k):
            self.k = k
        def get_relevant_documents(self, query):
            return []
        def invoke(self, query):
            return self.get_relevant_documents(query)
        # allow usage like: {"context": retriever | format_docs, ...}
        def __or__(self, fn):
            return lambda q: fn(self.get_relevant_documents(q))

    logging.warning("build_retriever: db is None, using empty retriever for tests.")
    return _EmptyRetriever(top_k)


def build_rag_pipeline(db, llm_model, top_k=6, temperature=0):
    """
    Build a lightweight RAG pipeline.
    - Works even when db=None (empty retriever).
    - Uses Dummy LLM if llm_model == 'dummy' or GROQ_API_KEY is missing.
    """
    retriever = build_retriever(db=db, top_k=top_k)

    if llm_model == "dummy" or ChatGroq is None or not _has_groq_key():
        llm = _DummyLLM()
    else:
        llm = ChatGroq(model=llm_model, temperature=temperature)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using the provided context:\n\n{context}\n\nQuestion: {question}"
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("RAG pipeline successfully built.")
    return rag_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def build_and_save_faiss(data_path, db_path, embedding_model, chunk_size=500, overlap=100):
    logging.info("Building FAISS index from processed data...")

    with open(data_path, "rb") as f:
        docs = pickle.load(f)

    # 如果是 prepare 输出的 list[dict]，转回 Document
    if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], dict) and "page_content" in docs[0]:
        docs = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in docs]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(db_path)

    logging.info(f"FAISS index saved to {db_path}")
    return db

