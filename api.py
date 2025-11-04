from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import build_retriever, build_llm, answer_with_policy

app = FastAPI()
retriever = build_retriever(k=6)
llm = build_llm()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    out = answer_with_policy(q.question, retriever=retriever, llm=llm)
    return out  # {"answer":..., "citations":[...], "policy":"ok|refuse"}

