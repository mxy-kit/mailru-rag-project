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
