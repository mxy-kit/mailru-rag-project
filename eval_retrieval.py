# eval_retrieval.py
from rag_pipeline import build_retriever
from typing import List, Tuple
import json, numpy as np

def recall_mrr(queries: List[str], gold_urls: List[List[str]], k=6)->Tuple[float,float]:
    retriever = build_retriever(k=k)
    hits, rr = [], []
    for q, gold in zip(queries, gold_urls):
        docs = retriever.get_relevant_documents(q)
        urls = [d.metadata.get("url","") for d in docs]
        hit = any(u in urls for u in gold)
        hits.append(hit)
        rank = min((urls.index(u)+1 for u in gold if u in urls), default=0)
        rr.append(0 if rank==0 else 1/rank)
    return np.mean(hits), np.mean(rr)

if __name__ == "__main__":
    data = json.load(open("mini_eval.json","r"))
    rec, mrr = recall_mrr([x["q"] for x in data], [x["gold"] for x in data], k=6)
    print(f"Recall@6={rec:.3f}  MRR@6={mrr:.3f}")
