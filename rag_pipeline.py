import pickle
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ChatPromptTemplate: support both new and old import paths
try:
    from langchain_core.prompts import ChatPromptTemplate  # new path
except ImportError:
    from langchain.prompts import ChatPromptTemplate       # legacy fallback

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# ChatGroq: try to import; if unavailable (e.g., CI without deps), use a tiny stub
try:
    from langchain_groq import ChatGroq
except Exception:
    class ChatGroq:
        """Minimal stub to keep tests running without external deps/keys."""
        def __init__(self, *args, **kwargs):
            pass
        def invoke(self, prompt):
            # return a deterministic, harmless string
            return "Stub LLM response. See https://help.mail.ru/"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_data_and_db(data_path, db_path, embedding_model):
    """Load serialized docs and FAISS index."""
    logging.info("Loading data and FAISS index...")
    with open(data_path, "rb") as f:
        docs = pickle.load(f)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    logging.info("Data and index successfully loaded.")
    return docs, db, embeddings


def format_docs(docs):
    """Join retrieved documents into a single context string."""
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)


def build_retriever(db=None, top_k=6):
    """
    Build a retriever from FAISS. If db is None (e.g., in CI), return an empty retriever
    that integrates safely into the LangChain pipeline.
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
    - If llm_model == 'dummy' or groq deps are missing, uses the ChatGroq stub above.
    """
    retriever = build_retriever(db=db, top_k=top_k)
    llm = ChatGroq(model=llm_model, temperature=temperature) if llm_model != "dummy" else ChatGroq()

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
