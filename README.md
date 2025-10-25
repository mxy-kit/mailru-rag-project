
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

| File                                   | Purpose                                                                                                                                                                     |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`rag_mailru_qa_with_outputs.ipynb`** | The full experimental notebook with all code, outputs, and visualizations. This is the main reference for reproducing results and evaluation metrics.                       |
| **`rag_pipeline.py`**                  | Contains reusable functions for data loading, preprocessing, embedding generation, and RAG pipeline construction.                                                           |
| **`train.py`**                         | Serves as the entry point for training and evaluation; reads parameters from `config.yaml`, initializes embeddings, builds the retriever–generator chain, and logs results. |
| **`test_pipeline.py`**                 | Includes lightweight tests to validate data integrity, pipeline structure, and output types — not model accuracy. Used to simulate CI/CD integration.                       |
| **`config.yaml`**                      | Stores configuration parameters such as model names, data paths, random seed, and retriever settings for reproducibility.                                                   |
| **`requirements.txt`**                 | Lists core dependencies (LangChain, FAISS, HuggingFace, Torch, etc.) required to reproduce the pipeline.                                                                    |
| **`.github/workflows/test.yml`**       | Automates unit testing via GitHub Actions upon each commit (verifies that the repository is self-consistent).                                                               |

### Note:
- The .py scripts represent modularized components of the same pipeline shown in the notebook.
- They are designed for MLOps compliance and structural clarity, rather than independent execution.



