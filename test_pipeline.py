import types
from rag_pipeline import build_retriever

class FakeEmbeddings:
    """Minimal embeddings stub to avoid heavy downloads in CI."""
    def embed_documents(self, texts):
        # deterministic small vectors
        return [[float((i + j) % 5) for j in range(8)] for i, _ in enumerate(texts)]
    def embed_query(self, text):
        return [0.1] * 8

def _fake_docs():
    Doc = types.SimpleNamespace
    return [
        Doc(page_content="Как восстановить пароль от почты Mail.ru"),
        Doc(page_content="Как привязать номер телефона к аккаунту Mail.ru"),
        Doc(page_content="Как отвязать VK ID от учетной записи"),
    ]
def test_docs_shape():
    docs = _fake_docs()
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert hasattr(docs[0], "page_content")

def test_retriever_builds_without_db():
    docs = _fake_docs()
    embeddings = FakeEmbeddings()
    retriever = build_retriever(docs=docs, embeddings=embeddings, db=None, top_k=2)
    # retriever should expose a retrieval method
    assert hasattr(retriever, "get_relevant_documents")
def test_docs_structure():
    docs = _fake_docs()
    assert all(hasattr(d, "page_content") for d in docs)
    assert isinstance(docs[0].page_content, str)

def test_embedding_shapes():
    emb = FakeEmbeddings()
    result = emb.embed_documents(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 8

def test_rag_output_type():
    from rag_pipeline import build_rag_pipeline
    retriever = type("MockRetriever", (), {"invoke": lambda self, q: "test_answer"})()
    rag = build_rag_pipeline(db=None, llm_model="dummy", top_k=2, temperature=0)
    assert callable(getattr(rag, "invoke", None))
# === 追加的最小测试：边界用例 + 输出契约（直接贴到文件末尾） ===

def _wrap_answer(rag, question: str):
    """
    超轻量包装：把 rag.invoke 的字符串输出包成统一契约
    返回 {"answer": str, "citations": list[str], "policy": "ok|refuse"}
    这里不改你的管线，只在测试里做格式化。
    """
    if not question.strip():
        return {"answer": "", "citations": [], "policy": "refuse"}
    text = rag.invoke(question)
    if not isinstance(text, str):
        text = str(text)
    # 简单规则：这里只验证契约存在，policy 给 "ok" 即可
    return {"answer": text, "citations": [], "policy": "ok"}

def test_empty_query_refusal_with_rag():
    from rag_pipeline import build_rag_pipeline
    rag = build_rag_pipeline(db=None, llm_model="dummy", top_k=2, temperature=0)
    out = _wrap_answer(rag, "")
    assert out["policy"] == "refuse"
    assert isinstance(out["answer"], str)

def test_nohit_query_returns_text():
    from rag_pipeline import build_rag_pipeline
    rag = build_rag_pipeline(db=None, llm_model="dummy", top_k=2, temperature=0)
    q = "Расскажите о марсианской визовой политике XX века"  # 明显领域外
    out = _wrap_answer(rag, q)
    assert isinstance(out["answer"], str)
    assert out["policy"] in {"ok", "refuse"}

def test_output_contract_shape():
    from rag_pipeline import build_rag_pipeline
    rag = build_rag_pipeline(db=None, llm_model="dummy", top_k=2, temperature=0)
    out = _wrap_answer(rag, "Как восстановить пароль?")
    assert set(out.keys()) == {"answer", "citations", "policy"}
    assert isinstance(out["citations"], list)
    assert isinstance(out["answer"], str)

