
# Retrieval-Augmented Generation (RAG) for Mail.ru Help Center  


---
# Mail.ru Help Center RAG + MLOps (DVC + MLflow + Docker + TorchServe)

## ✅ Homework Checklist (MLOps)

- [x] **Task 1 (DVC)**: data/model artifacts are tracked by DVC + remote storage configured; reproducible via `dvc pull && dvc repro`
- [x] **Task 2 (MLflow)**: every `python train.py` creates a new MLflow run with params/metrics/artifacts (+ DVC hash tags)
- [x] **Task 3 (Docker offline inference)**: reproducible image builds and runs `src/predict.py` with `--input_path/--output_path`
- [x] **Task 4 (TorchServe online service)**: Docker image starts TorchServe and serves `/predictions/mymodel`
- [x] **CI**: GitHub Actions runs tests and builds docker image

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

### Note:
- The .py scripts represent modularized components of the same pipeline shown in the notebook.
- They are designed for MLOps compliance and structural clarity, rather than independent execution.



## Task 1 — DVC: Data/Model Versioning

### What is tracked by DVC
- Raw dataset (large file): `data/raw/help_mail_ru.pkl` (tracked as a `.dvc` pointer file, so the big file is stored outside Git)
- Pipeline outputs (examples, as declared in `dvc.yaml`): `models/`, `db/` (FAISS index), etc.

> Large files are NOT stored in Git. Git stores only small `.dvc` pointer files + `dvc.yaml` / `dvc.lock` to reproduce exact versions.

### Where the data/models physically live (remote storage)
- DVC remote storage: **Google Drive folder**  
  https://drive.google.com/drive/u/1/folders/1pqyGYExEy1bYDGlVsA-5ve3KTgCpOflt

### Reproduce everything (fresh clone)
```bash
git clone https://github.com/mxy-kit/mailru-rag-project.git
cd mailru-rag-project
git checkout hw2_dvc_mlflow_docker_torchserve

pip install -r requirements.txt

# download all DVC-tracked data/models from Google Drive
dvc pull

# run the full pipeline: prepare -> train -> evaluate
dvc repro
```
Verify remote configuration 
```bash
dvc remote list
dvc remote list --verbose
```

```md
Pipeline stages are defined in `dvc.yaml` (prepare/train/evaluate) and the exact artifact versions are locked in `dvc.lock`.
```
**`prepare`**: preprocess raw corpus → outputs processed artifacts
**`train`**: (optional) finetune embedder + build FAISS index → outputs **`models/ `**and **`db/`**
**`evaluate`**: compute metrics → outputs **`metrics/... `**and **`reports/...`**

## Task 2 — MLflow: Experiment Tracking (with DVC linkage)

### What is tracked in MLflow
Each run of `python train.py ...` creates a separate MLflow run that logs:
- **Parameters**: seed, chunk_size, overlap, retrieval_top_k, temperature, embedding_model, data_path, db_path, etc.
- **Metrics**: e.g. `train_total_seconds` (and other evaluation metrics if you log them in `evaluate.py`)
- **Artifacts**:
  - `dvc.lock` (to bind experiment ↔ exact data/model versions)
  - fine-tuned embedding model directory (logged under artifacts)
  - FAISS index folder (`db/`) (logged under artifacts)
  - MLflow Model (PyFunc) for SentenceTransformer embeddings (so it can appear in MLflow UI “Models”)

### DVC + MLflow binding 
This project links DVC versions to MLflow runs by:
- logging `dvc.lock` as an MLflow artifact
- setting tags with DVC hashes, e.g.:
  - `dvc_raw_md5` (read from `data/raw/help_mail_ru.pkl.dvc`)
  - `dvc_lock_sha1` (hash of `dvc.lock`)

### Where the data/models physically live (remote storage)
- DVC remote storage: **Google Drive folder**  
  https://drive.google.com/drive/u/1/folders/1pqyGYExEy1bYDGlVsA-5ve3KTgCpOflt
### How to run MLflow locally (UI)
This project uses **local MLflow backend**.

1) Run training (creates an MLflow run):
```bash
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
dvc repro
```
 or directly:
 ```bash
python train.py --config config.yaml
```
Start MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open in browser:

http://127.0.0.1:5000

In the UI you should see the experiment mailru-rag and multiple runs.
Inside each run you can find parameters, metrics, and artifacts (including dvc.lock and model/index artifacts).


# Task 3 — Docker: Offline Inference Image

This document describes how to build and run a reproducible Docker image for **offline inference**.

---

## DockerHub image (Task 3)

The **offline inference** image is published to DockerHub:

https://hub.docker.com/repository/docker/2700264072/mailru-rag-offline/general

Image tag used in this homework:
- `2700264072/mailru-rag-offline:v1`

---

## What the container does

When the container starts, it runs `src/predict.py`, which:

1. loads the required artifacts from disk (e.g., FAISS index / embedding model dir, depending on your config)
2. accepts command-line arguments:
   - `--input_path`: path to input file (inside container)
   - `--output_path`: path to save predictions (inside container)
3. reads queries from `input_path`, performs inference, and writes results to `output_path`

**Input format:** CSV must contain a column named `query`.

**Output example:** `outputs/preds.csv`

---

## Prerequisites

- Docker Desktop installed

(Optional) DVC installed if you want to pull artifacts via DVC:
```bash
pip install "dvc[gdrive]"
```
```bash
pip install -r requirements.txt
```
## If your model/index artifacts are tracked by DVC:
```bash
dvc pull
```
##Run offline inference
```bash
docker build -t ml-app:v1 .
docker run --rm `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\outputs:/app/outputs `
  ml-app:v1 `
  --input_path /app/data/sample_input.csv `
  --output_path /app/outputs/preds.csv
```
##Notes

--.dockerignore is used to exclude unnecessary files from the build context.

--Large artifacts should not be committed to Git; use DVC (dvc pull) to restore them when needed.


---



# Task 4 — TorchServe: Online Service in Docker

This document describes how to run the model as an **online REST service** using TorchServe.

---

## DockerHub image

The TorchServe service image is published to DockerHub:

https://hub.docker.com/repository/docker/2700264072/mymodel-serve/general

Image tag used in this homework:
- `2700264072/mymodel-serve:v1`

---

## What is included

- Base image: `pytorch/torchserve`
- Model archive: `model-store/mymodel.mar` (built via `torch-model-archiver`)
- Custom inference handler: `handler.py` (preprocessing + postprocessing)
- Container startup automatically:
  - starts TorchServe
  - registers the model under the name `mymodel`

---

## Run the service locally

### 1) Pull from DockerHub

```bash
docker pull 2700264072/mymodel-serve:v1
```

### 2) Run container

```bash
docker run -d --name mymodel-serve \
  -p 8080:8080 -p 8081:8081 \
  2700264072/mymodel-serve:v1
```
## Inference Interface

```bash
curl -s http://localhost:8081/ping
curl -s http://localhost:8081/models
```

### 3)Example REST request
```bash
curl -X POST http://localhost:8080/predictions/mymodel -T sample_input.json

```
### 4)Example response (truncated)
```bash
[ { "query": "{query:как восстановить пароль?}", "top_k": 6, "results": [ { "score": 6.5856781005859375, "preview": "Восстановить доступ ...", "metadata": { "source": "https://help.mail.ru/mail/security/", "title": "Безопасность — Почта Mail ...", "language": "ru-RU" } }, { "score": 9.758225440979004, "preview": "почту взломали ... Измените пароль ...", "metadata": { "source": "https://help.mail.ru/mail/security/restore/blocked/", "title": "Как войти в Почту Mail, если забыли пароль ...", "language": "ru-RU" } } ] } ]

```
