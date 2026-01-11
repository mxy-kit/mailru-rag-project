import json
import os
from typing import Any, List, Union

from ts.torch_handler.base_handler import BaseHandler
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class MailruRagHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.top_k = 6
        self.db = None
        self.emb = None
        self.st_model = None

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # 你的 .mar 里文件在根目录，所以都用 model_dir
        embeddings_dir = model_dir
        db_dir = model_dir

        self.top_k = int(os.environ.get("TOP_K", "6"))

        # 加载 sentence-transformers 模型（从 model_dir）
        self.st_model = SentenceTransformer(embeddings_dir)

        # LangChain 的 Embeddings（同样指向 model_dir）
        self.emb = HuggingFaceEmbeddings(model_name=embeddings_dir)

        # index.faiss / index.pkl 在 mar 根目录（FAISS.load_local 会在 db_dir 下找）
        self.db = FAISS.load_local(
            db_dir,
            self.emb,
            allow_dangerous_deserialization=True
        )

        self.initialized = True

    def preprocess(self, data):
        import json
        # TorchServe gives a list of requests
        item = data[0] if isinstance(data, list) and data else data
        payload = None
        if isinstance(item, dict):
            payload = item.get("body") if "body" in item else item.get("data")
            if payload is None:
                payload = item
        else:
            payload = item

        # bytes -> str
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", errors="ignore")

        obj = payload
        # str -> try json
        if isinstance(payload, str):
            try:
                obj = json.loads(payload)
            except Exception:
                obj = payload

        # dict/list -> extract query
        if isinstance(obj, dict):
            q = obj.get("query") or obj.get("data") or obj.get("text") or obj.get("question")
            if q is None:
                q = json.dumps(obj, ensure_ascii=False)
        elif isinstance(obj, list):
            q = obj[0] if obj else ""
        else:
            q = str(obj)

        return q


    def inference(self, queries, *args, **kwargs):
        """
        保持你原来返回格式：List[{"query":..., "top_k":..., "results":[...]}]
        同时兼容 preprocess 返回单个 str。
        """
        # 兼容 preprocess 返回 str，避免按字符遍历
        if isinstance(queries, str):
            queries = [queries]
        elif not isinstance(queries, list):
            queries = [str(queries)]

        results = []
        for q in queries:
            docs_with_scores = self.db.similarity_search_with_score(q, k=self.top_k)
            items = []
            for doc, score in docs_with_scores:
                text = (doc.page_content or "").replace("\n", " ").strip()
                items.append(
                    {
                        "score": float(score),
                        "preview": text[:200],
                        "metadata": doc.metadata or {},
                    }
                )
            results.append({"query": q, "top_k": self.top_k, "results": items})
        return results

    def postprocess(self, inference_output: Any):
        return [json.dumps(inference_output, ensure_ascii=False)]
