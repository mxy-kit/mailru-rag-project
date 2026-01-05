import argparse
import json
import pickle
import random
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)          # data/processed/help_mail_ru.pkl
    ap.add_argument("--db", required=True)            # db
    ap.add_argument("--embedding_model", required=True)  # models/fine_tuned_embeddings or hf name
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", required=True)           # metrics/retrieval.json
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=50)      # how many random queries to test
    args = ap.parse_args()

    random.seed(args.seed)

    # load docs
    with open(args.data, "rb") as f:
        docs = pickle.load(f)

    if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], dict) and "page_content" in docs[0]:
        docs = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in docs]

    # simple synthetic eval:
    # query = random chunk of a doc, relevant = the same doc id (approx)
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    db = FAISS.load_local(args.db, embeddings, allow_dangerous_deserialization=True)

    n = min(args.n, len(docs))
    sample = random.sample(docs, n)

    hits = 0
    for d in sample:
        text = d.page_content
        if not text:
            continue
        # take a short snippet as query
        q = text[:200]
        retrieved = db.similarity_search(q, k=args.k)
        # hit if any retrieved doc has high overlap with original snippet (cheap heuristic)
        ok = any((q[:80] in r.page_content) for r in retrieved if r.page_content)
        hits += int(ok)

    hit_at_k = hits / n if n else 0.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"hit_at_k": hit_at_k, "k": args.k, "n": n}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] saved metrics to {out_path} : hit@{args.k}={hit_at_k:.3f}")


if __name__ == "__main__":
    main()
