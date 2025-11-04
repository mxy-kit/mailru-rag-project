
# Retrieval-Augmented Generation (RAG) for Mail.ru Help Center  


---

##  Project Goal 

The goal of this project is to develop an intelligent assistant for the Mail.ru Help Center using the **Retrieval-Augmented Generation (RAG)** approach.  
The system retrieves relevant documents from the help knowledge base and generates concise, human-like answers in Russian.  


##  Business Motivation 

Many FAQ systems contain thousands of articles, but users expect **fast and accurate** answers written in natural language.  
The RAG model allows combining external document retrieval with generative language modeling, enabling the assistant to answer questions even if the facts are not in its weights.  



## Target Metrics 

| Metric | Target | Meaning |
|---------|---------|---------|
| BLEU score | ≥ 0.04 | Generation similarity to reference |
| LLM-as-judge | ≥ 3.5 / 5 | Semantic answer quality |
| Embedding uniformity | ≤ -1.7 | Balanced vector space |
| Alignment ratio | ≤ 0.03 | Stable embedding representation |

## Production Metrics (Targets)
- Latency: P50 ≤ 200 ms, P95 ≤ 500 ms
- Error rate: ≤ 1%
- Resource: CPU ≤ 2 vCPU, RAM ≤ 2 GB (single instance)
- Availability: ≥ 99.5%
- Business KPI: One-shot answer rate ≥ 70%, Escalation to human ≤ 15%





## Pipeline Description

###  Data Collection and Preprocessing
- Source: [https://help.mail.ru](https://help.mail.ru)
- Tools: `RecursiveUrlLoader` (LangChain) + `BeautifulSoup`
- Unnecessary UI elements, navigation links, and survey blocks were removed with regex filtering.  
- Data were serialized and stored as `help_mail_ru.pkl` for reproducibility.

### Embedding and Indexing
- Models evaluated:  
  `deepvk/USER-bge-m3`, `deepvk/USER-base`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Vector database: **FAISS**, persisted locally (`db/`)
- Chunking: size = 500, overlap = 100  
- MiniLM demonstrated the best uniformity (−2.41) and alignment (0.028), thus selected as the final embedder.

### Fine-tuning Step

To further adapt the embedding model to the Mail.ru Help Center domain, a lightweight fine-tuning procedure was implemented in `train.py` before building the FAISS index.  
The multilingual MiniLM model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) was fine-tuned on several representative query–answer pairs extracted from the help corpus.

This step slightly adjusts the embedding space, improving retrieval precision for domain-specific phrasing (e.g., “восстановить пароль”, “удалить аккаунт”).


The fine-tuned model is automatically trained and saved before the RAG pipeline initialization.  
This ensures the system uses embeddings better aligned with the Mail.ru support domain.

My  **model weights** have been uploaded to Hugging Face for reproducibility:  
[[ https://huggingface.co/xinyuema/mailru-finetuned-minilm](https://huggingface.co/xinyuema/mailru-finetuned-minilm)] 


### Retrieval and Generation
- Retriever: `FAISS.as_retriever(k=6)`  
- LLM: `llama-3.1-8b-instant` via **Groq API**
- Prompt design:
  - **Without RAG** – refusal policy for unrelated questions (prevents hallucination)
  - **With RAG** – answer strictly within retrieved context  
- Implemented through LangChain runnable composition:
```python
 rag = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Persistence

- Data `help_mail_ru.pkl` and FAISS index (`db/`) allow full reproducibility.

- No weight saving (save_pretrained) is required since pretrained API models are used.

### Experiment Plan
- **Embedding ablations:** MiniLM vs USER-bge; `chunk_size ∈ {300, 500}`, `retrieval_top_k ∈ {4, 6, 8}`.
- **Retrieval metrics:** Recall@k and MRR (finetuned vs. base).
- **End metrics:** Target BLEU ≥ 0.04 and LLM-as-judge ≥ 3.5; stop when both are met.
- **Hallucination control:** refusal policy when context is empty or out-of-scope.


### Key Results

| Embedding Model | Uniformity | Alignment | Comment                       |
| --------------- | ---------- | --------- | ----------------------------- |
| USER-bge-m3     | −1.92      | 0.095     | baseline                      |
| USER-base       | −2.15      | 0.065     | good                          |
| MiniLM multi    | **−2.41**  | **0.028** | best for semantic retrieval |

- The MiniLM model achieved the most uniform and stable embedding space, improving retrieval precision in the RAG pipeline.

###  Evaluation Summary

| Metric               | Target    | Actual                 | Status           | Meaning                         |
| -------------------- | --------- | ---------------------- | ---------------- | ------------------------------- |
| BLEU score           | ≥ 0.04    | 0.0459                 | ✅                | Similarity to reference answers |
| LLM-as-judge         | ≥ 3.5 / 5 | 3.5                    | ✅                | Semantic quality of answers     |
| Embedding uniformity | ≤ −1.7    | −2.41                  | ✅                | Balanced vector distribution    |
| Alignment ratio      | ≤ 0.03    | 0.028                  | ✅                | Stable semantic representation  |

### Configuration and Execution

All runtime parameters are stored in `config.yaml`:

```yaml
seed: 42
chunk_size: 500
embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
llm_model: "llama-3.1-8b-instant"
retrieval_top_k: 6
temperature: 0
```
To run the project:

```bash
pip install -r requirements.txt
python train.py
```
Logging is handled through Python’s logging module.
Each stage (loading → retrieval → generation) prints structured messages.

### Reproducibility
- All random seeds are fixed in `train.py` (Python/NumPy/PyTorch).
- Data (`help_mail_ru.pkl`) and FAISS index (`db/`) are persisted and reused.
- Finetuned embeddings are saved with `SentenceTransformer.save_pretrained()` (HF-compatible).

### CLI
```bash
pip install -r requirements.txt
# Optional: use local finetuned embeddings
unzip fine_tuned_embeddings.zip -d ./fine_tuned_embeddings
python train.py --config config.yaml --verbose


### Repository Structure Overview

- This repository follows a modular MLOps-style organization.
- Each Python file represents a logical component of the RAG pipeline:

| File                            | Purpose                                                                                                                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`README.md`**                 | Project overview, production targets, how to run, and evaluation results.                                                                                                     |
| **`rag_mailru_qa.ipynb`**       | End-to-end experimental notebook with code, outputs, and visualizations (reference for reproducing metrics).                                                                  |
| **`rag_pipeline.py`**           | Modular RAG components: data loading/cleaning, embedding & FAISS index, retriever → generator pipeline.                                                                       |
| **`train.py`**                  | Entry point: reads `config.yaml`, sets seed, (optionally) fine-tunes embeddings, builds the RAG pipeline, logs stages.                                                        |
| **`test_pipeline.py`**          | Lightweight tests for data structure, pipeline wiring, and output contract (not model accuracy). Runs in CI.                                                                  |
| **`data_validation.py`**        | Pandera schema + basic stats to validate the cleaned corpus (`help_mail_ru.pkl`).                                                                                             |
| **`eval_retrieval.py`**         | Retrieval evaluation (e.g., Recall@k / MRR) and small ablations for k/chunking/finetune vs. base.                                                                             |
| **`api.py`**                    | Optional FastAPI endpoint (`POST /ask`) returning `{answer, citations, policy}` for quick API integration.                                                                    |
| **`config.yaml`**               | Single source of truth for seed, models/paths, chunk size, `retrieval_top_k`, temperature, etc.                                                                               |
| **`requirements.txt`**          | Pinned core deps (LangChain, FAISS, Hugging Face, Torch, Pandera, PyTest, etc.).                                                                                              |
| **`.github/workflows/*.yml`**   | GitHub Actions: install deps, run lint/tests on every push/PR; fails on errors.                                                                                               |
| **`help_mail_ru.pkl`**          | Serialized, cleaned Mail.ru Help Center corpus (recursive crawl + HTML parsing).                                                                                              |
| **`db/`**                       | Persisted FAISS index (`index.faiss`, `index.pkl`) to avoid re-indexing.                                                                                                      |
| **`fine_tuned_embeddings.zip`** | Zipped Sentence-Transformers model after fine-tuning (MiniLM). **Unzip to `./fine_tuned_embeddings/`** and set `embedding_model: "./fine_tuned_embeddings"` in `config.yaml`. |

### Note:
- The .py scripts represent modularized components of the same pipeline shown in the notebook.
- They are designed for MLOps compliance and structural clarity, rather than independent execution.



