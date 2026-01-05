import argparse
import yaml
import logging
import random
import numpy as np
import os
import torch

from rag_pipeline import build_and_save_faiss

from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    tuned_model_path = fine_tune_model(
        base_model=config["embedding_model"],
        output_dir=config["model_dir"],
    )

    build_and_save_faiss(
    data_path=config["data_path"],
    db_path=config["db_path"],
    embedding_model=tuned_model_path,
    chunk_size=int(config.get("chunk_size", 500)),
    overlap=int(config.get("overlap", 100)),
)


    logging.info("Train stage finished: embeddings + FAISS index are ready.")


if __name__ == "__main__":
    main()
