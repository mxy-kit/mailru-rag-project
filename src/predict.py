import argparse

import csv
import json
from pathlib import Path
from typing import List, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def read_queries(input_path: Path) -> List[str]:
    """Support: .txt (one query per line), .json (list[str]), .csv (column=query)."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".txt":
        lines = [l.strip() for l in input_path.read_text(encoding="utf-8").splitlines()]
        return [l for l in lines if l]

    if suffix == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError("JSON input must be a list of strings, e.g. [\"q1\", \"q2\"]")
        return [x.strip() for x in data if x.strip()]

    if suffix == ".csv":
        # expects header with column named "query"
        with input_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "query" not in reader.fieldnames:
                raise ValueError("CSV input must have a 'query' column")
            queries = []
            for row in reader:
                q = (row.get("query") or "").strip()
                if q:
                    queries.append(q)
            return queries

    raise ValueError("Unsupported input format. Use .txt / .json / .csv")


def retrieve_top1(db: FAISS, query: str) -> Tuple[str, float]:
    """Return (top_doc_preview, score)."""
    # similarity_search_with_score returns List[Tuple[Document, float]]
    res = db.similarity_search_with_score(query, k=1)
    if not res:
        return "", float("nan")
    doc, score = res[0]
    text = (doc.page_content or "").replace("\n", " ").strip()
    preview = text[:300]
    return preview, float(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="Path to queries (.txt/.json/.csv)")
    ap.add_argument("--output_path", required=True, help="Path to write preds.csv")

    # optional (nice for docker)
    ap.add_argument("--db_path", default="db", help="FAISS db directory (default: db)")
    ap.add_argument(
        "--embedding_model",
        default="models/fine_tuned_embeddings",
        help="HF model name or local path (default: models/fine_tuned_embeddings)",
    )
    args = ap.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    queries = read_queries(input_path)
    if not queries:
        raise ValueError("No queries found in input.")

    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    db = FAISS.load_local(args.db_path, embeddings, allow_dangerous_deserialization=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "top_doc_preview", "score"])
        writer.writeheader()

        for q in queries:
            preview, score = retrieve_top1(db, q)
            writer.writerow({"query": q, "top_doc_preview": preview, "score": score})

    print(f"[OK] wrote predictions: {output_path} (rows={len(queries)})")


if __name__ == "__main__":
    main()
