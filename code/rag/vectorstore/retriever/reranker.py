from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .config import RERANKER_MODEL, TOP_K_INITIAL, TOP_K_FINAL


@dataclass
class RerankResult:
    id: str
    content: str
    metadata: Dict[str, Any]
    rerank_score: float
    original_rank: int


class Reranker:
    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        top_k_initial: int = TOP_K_INITIAL,
        top_k_final: int = TOP_K_FINAL
    ):
        self.model_name = model_name
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final
        self.model = None
        self._load_model()

    def _load_model(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers 未安装。"
                "请运行: pip install sentence-transformers"
            )
        print(f"正在加载重排序模型: {self.model_name}")
        self.model = CrossEncoder(self.model_name, max_length=512)
        print("重排序模型加载完成")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[RerankResult]:
        if top_k is None:
            top_k = self.top_k_final

        if not documents:
            return []

        doc_texts = [doc['content'] if isinstance(doc, dict) else doc for doc in documents]
        doc_ids = [doc['id'] if isinstance(doc, dict) else str(i) for i, doc in enumerate(documents)]
        doc_metadatas = [doc['metadata'] if isinstance(doc, dict) else {} for doc in documents]

        pairs = [[query, doc_text] for doc_text in doc_texts]
        scores = self.model.predict(pairs)

        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        if isinstance(scores, float):
            scores = [scores]

        results_with_scores = [
            (doc_id, doc_text, doc_metadata, score, original_rank)
            for doc_id, doc_text, doc_metadata, score, original_rank
            in zip(doc_ids, doc_texts, doc_metadatas, scores, range(len(documents)))
        ]

        results_with_scores.sort(key=lambda x: x[3], reverse=True)

        return [
            RerankResult(
                id=item[0],
                content=item[1],
                metadata=item[2],
                rerank_score=float(item[3]),
                original_rank=item[4]
            )
            for item in results_with_scores[:top_k]
        ]

    def rerank_with_initial_results(
        self,
        query: str,
        initial_results: List,
        top_k: int = None
    ) -> List[RerankResult]:
        if top_k is None:
            top_k = self.top_k_final

        docs_for_rerank = [
            {
                'id': result.id,
                'content': result.content,
                'metadata': result.metadata
            }
            for result in initial_results
        ]

        return self.rerank(query, docs_for_rerank, top_k)
