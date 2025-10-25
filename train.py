
import yaml
import time
import logging
from rag_pipeline import load_data_and_db, build_rag_pipeline

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def main():
    logging.info("Starting RAG Mail.ru assistant...")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    docs, db, embeddings = load_data_and_db(
        config["data_path"], config["db_path"], config["embedding_model"]
    )

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

    print("\n🧠 Question:", query)
    print("💬 Answer:", answer)
    print(f"⚡ Response latency: {latency:.3f} s")

if __name__ == "__main__":
    main()
