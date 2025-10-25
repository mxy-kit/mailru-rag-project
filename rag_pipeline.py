
import pickle
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def load_data_and_db(data_path, db_path, embedding_model):
    logging.info("Loading data and FAISS index...")
    with open(data_path, "rb") as f:
        docs = pickle.load(f)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    logging.info("Data and index successfully loaded.")
    return docs, db, embeddings

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_rag_pipeline(db, llm_model, top_k=6, temperature=0):
    retriever = db.as_retriever(search_kwargs={"k": top_k})
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
