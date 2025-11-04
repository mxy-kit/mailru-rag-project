import yaml
import time
import logging
from rag_pipeline import load_data_and_db, build_rag_pipeline
import random, numpy as np
import os
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def fine_tune_model(output_dir="/content/drive/MyDrive/fine_tuned_embeddings/"):
    """
    Fine-tune the multilingual MiniLM model on a small Mail.ru Help Center dataset.
    """
    os.environ["WANDB_MODE"] = "disabled"

    logging.info("Starting fine-tuning of SentenceTransformer...")

    # --- Step 1: Prepare tiny domain-specific samples ---
    train_examples = [
        InputExample(texts=["Как восстановить пароль?", "Забытый пароль, восстановление"]),
        InputExample(texts=["Удалить аккаунт", "Как удалить свой профиль?"]),
        InputExample(texts=["Где найти настройки?", "Настройки аккаунта Mail.ru"])
    ]
    train_dataloader = DataLoader(train_examples, batch_size=2, shuffle=True)

    # --- Step 2: Load pretrained model ---
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # --- Step 3: Define contrastive loss ---
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # --- Step 4: Train for one epoch ---
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        show_progress_bar=True
    )

    # --- Step 5: Save fine-tuned model ---
    model.save(output_dir)
    logging.info(f" Fine-tuning complete. Model saved to {output_dir}")
    return output_dir


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Starting RAG Mail.ru assistant...")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 固定随机种子
    set_seed(config["seed"])

    # --- Fine-tune model first ---
    tuned_model_path = fine_tune_model()

    # --- Load docs, DB, and embeddings ---
    docs, db, embeddings = load_data_and_db(
        config["data_path"],
        config["db_path"],
        tuned_model_path  # 使用微调后的 embedding 模型
    )

    # --- Build and run the RAG pipeline ---
    rag = build_rag_pipeline(
        db=db,
        llm_model=config["llm_model"],
        top_k=config["retrieval_top_k"],
        temperature=config["temperature"],
    )

    query = "Как отвязать VKID от почты?"
    start = time.time()
    answer = rag.invoke(query)
    latency = time.time() - start

    print("\nQuestion:", query)
    print("Answer:", answer)
    print(f"Response latency: {latency:.3f} s")


if __name__ == "__main__":
    main()
