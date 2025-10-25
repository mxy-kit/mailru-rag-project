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

def test_retriever_builds_without_db():
    docs = _fake_docs()
    embeddings = FakeEmbeddings()
    retriever = build_retriever(docs=docs, embeddings=embeddings, db=None, top_k=2)
    # retriever should expose a retrieval method
    assert hasattr(retriever, "get_relevant_documents")
