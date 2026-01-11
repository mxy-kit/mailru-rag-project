import argparse
import yaml
import logging
import random
import numpy as np
import os
import time
import hashlib

import torch

import mlflow
import mlflow.pyfunc

from rag_pipeline import build_and_save_faiss

from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader


REGISTERED_MODEL_NAME = "mailru-rag-embeddings"
MLFLOW_ST_MODEL_ARTIFACT = "st_pyfunc_model"
RAW_DVC_FILE = "data/raw/help_mail_ru.pkl.dvc"   # 按你的实际路径改


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dvc_md5(dvc_file: str) -> str:
    # 读取 *.dvc 文件里的 md5，作为数据版本标识
    if not os.path.exists(dvc_file):
        return ""
    with open(dvc_file, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return str(y.get("md5", ""))


class STEmbeddingPyFunc(mlflow.pyfunc.PythonModel):
    """
    把 SentenceTransformer 打包成 MLflow PyFunc model：
    输入：DataFrame（列名 text）或 list[str]
    输出：np.ndarray embeddings
    """
    def load_context(self, context):
        self.model = SentenceTransformer(context.artifacts["st_model_dir"])

    def predict(self, context, model_input):
        # 尽量兼容：DataFrame / Series / list
        try:
            import pandas as pd
            if isinstance(model_input, pd.DataFrame):
                if "text" in model_input.columns:
                    texts = model_input["text"].astype(str).tolist()
                else:
                    texts = model_input.iloc[:, 0].astype(str).tolist()
            else:
                texts = list(model_input)
        except Exception:
            texts = list(model_input)

        emb = self.model.encode(texts, normalize_embeddings=True)
        return emb


def fine_tune_model(base_model: str, output_dir: str):
    os.environ["WANDB_MODE"] = "disabled"
    logging.info("Starting fine-tuning of SentenceTransformer...")

    train_examples = [
        InputExample(texts=["Как восстановить пароль?", "Забытый пароль, восстановление"]),
        InputExample(texts=["Удалить аккаунт", "Как удалить свой профиль?"]),
        InputExample(texts=["Где найти настройки?", "Настройки аккаунта Mail.ru"]),
    ]
    train_dataloader = DataLoader(train_examples, batch_size=2, shuffle=True)

    model = SentenceTransformer(base_model)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        show_progress_bar=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    logging.info(f"Fine-tuning complete. Model saved to {output_dir}")
    return output_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(int(config["seed"]))

    # ---- MLflow setup ----
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mailru-rag")

    with mlflow.start_run(run_name="train_finetune_faiss"):
        # params
        mlflow.log_params({
            "seed": int(config["seed"]),
            "chunk_size": int(config.get("chunk_size", 500)),
            "overlap": int(config.get("overlap", 100)),
            "retrieval_top_k": int(config.get("retrieval_top_k", 6)),
            "temperature": float(config.get("temperature", 0)),
            "embedding_model_base": config["embedding_model"],
            "llm_model": config.get("llm_model", ""),
            "data_path": config["data_path"],
            "db_path": config["db_path"],
        })

        # --- DVC linkage tags (加分项) ---
        dvc_raw_md5 = read_dvc_md5(RAW_DVC_FILE)
        if dvc_raw_md5:
            mlflow.set_tag("dvc_raw_md5", dvc_raw_md5)

        if os.path.exists("dvc.lock"):
            mlflow.set_tag("dvc_lock_sha1", sha1_file("dvc.lock"))

        start_total = time.time()

        # 1) finetune embeddings model
        tuned_model_path = fine_tune_model(
            base_model=config["embedding_model"],
            output_dir="models/fine_tuned_embeddings",
        )
        mlflow.log_param("tuned_model_path", tuned_model_path)

        # 2) build FAISS index and save to db/
        logging.info("Building FAISS index from processed data...")
        build_and_save_faiss(
            data_path=config["data_path"],
            db_path=config["db_path"],
            embedding_model=tuned_model_path,
            chunk_size=int(config.get("chunk_size", 500)),
            overlap=int(config.get("overlap", 100)),
        )

        total_seconds = time.time() - start_total
        mlflow.log_metric("train_total_seconds", float(total_seconds))

        # artifacts（你原来的都保留）
        if os.path.exists("dvc.lock"):
            mlflow.log_artifact("dvc.lock")
        if os.path.exists("models/fine_tuned_embeddings"):
            mlflow.log_artifacts("models/fine_tuned_embeddings", artifact_path="model_dir")
        if os.path.exists(config["db_path"]):
            mlflow.log_artifacts(config["db_path"], artifact_path="faiss_db")

        # ✅ 关键：把 SentenceTransformer 作为真正的 MLflow Model 记录
        # 这样 UI 的 Models 页面才能展示/注册版本
        mlflow.pyfunc.log_model(
            name=MLFLOW_ST_MODEL_ARTIFACT,
            python_model=STEmbeddingPyFunc(),
            artifacts={"st_model_dir": tuned_model_path},
        )

        # ✅ 自动注册到 Model Registry（Models 页面）
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{MLFLOW_ST_MODEL_ARTIFACT}"
        mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

        logging.info("Train stage finished: embeddings + FAISS index are ready.")


if __name__ == "__main__":
    main()
