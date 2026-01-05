import argparse
import json
import pickle
import random
from pathlib import Path
import hashlib
import yaml
import os

import mlflow
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


RAW_DVC_FILE = "data/raw/help_mail_ru.pkl.dvc"  # 按你的实际路径改


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dvc_md5(dvc_file: str) -> str:
    if not os.path.exists(dvc_file):
        return ""
    with open(dvc_file, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return str(y.get("md5", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)  # data/processed/help_mail_ru.pkl
    ap.add_argument("--db", required=True)  # db
    ap.add_argument("--embedding_model", required=True)  # models/fine_tuned_embeddings or hf name
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", required=True)  # metrics/retrieval.json
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=50)  # how many random queries to test
    args = ap.parse_args()

    # ---- MLflow setup FIRST ----
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mailru-rag")

    with mlflow.start_run(run_name="evaluate_retrieval"):
        # link DVC ↔ MLflow (tags)
        dvc_raw_md5 = read_dvc_md5(RAW_DVC_FILE)
        if dvc_raw_md5:
            mlflow.set_tag("dvc_raw_md5", dvc_raw_md5)

        if os.path.exists("dvc.lock"):
            mlflow.set_tag("dvc_lock_sha1", sha1_file("dvc.lock"))
            mlflow.log_artifact("dvc.lock")

        random.seed(args.seed)

        # --- load docs ---
        with open(args.data, "rb") as f:
            docs = pickle.load(f)

        # if stored as dicts -> convert to Document objects
        if (
            isinstance(docs, list)
            and len(docs) > 0
            and isinstance(docs[0], dict)
            and "page_content" in docs[0]
        ):
            docs = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in docs]

        # --- load FAISS ---
        embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
        db = FAISS.load_local(args.db, embeddings, allow_dangerous_deserialization=True)

        # --- simple synthetic eval ---
        n = min(args.n, len(docs))
        sample = random.sample(docs, n) if n > 0 else []

        hits = 0
        valid = 0

        for d in sample:
            text = getattr(d, "page_content", "") or ""
            text = text.strip()
            if not text:
                continue

            valid += 1
            q = text[:200]

            retrieved = db.similarity_search(q, k=args.k)

            q80 = q[:80]
            ok = any((q80 in (r.page_content or "")) for r in retrieved)
            hits += int(ok)

        hit_at_k = (hits / valid) if valid else 0.0

        # --- write metrics json ---
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "hit_at_k": hit_at_k,
                    "k": args.k,
                    "n": n,
                    "valid": valid,
                    "seed": args.seed,
                    "embedding_model": args.embedding_model,
                    "data_path": args.data,
                    "db_path": args.db,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[OK] saved metrics to {out_path} : hit@{args.k}={hit_at_k:.3f}")

        # --- MLflow logging ---
        mlflow.log_params(
            {
                "k": args.k,
                "n": n,
                "valid": valid,
                "seed": args.seed,
                "embedding_model": args.embedding_model,
                "data_path": args.data,
                "db_path": args.db,
                "out_path": str(out_path),
            }
        )
        mlflow.log_metric("hit_at_k", float(hit_at_k))  # 建议统一名字，方便对比
        mlflow.log_artifact(str(out_path))


if __name__ == "__main__":
    main()
