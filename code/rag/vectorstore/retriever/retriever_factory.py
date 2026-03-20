from enum import Enum
from typing import List, Optional
from .base_retriever import SearchResult
from .vector_retriever import VectorRetriever
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker, RerankResult
from .config import (
    ENABLE_HYBRID_SEARCH,
    ENABLE_RERANKER,
    TOP_K_VECTOR,
    TOP_K_INITIAL,
    TOP_K_FINAL
)


class RetrieverType(Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"


class RetrieverFactory:
    _vector_retriever: Optional[VectorRetriever] = None
    _hybrid_retriever: Optional[HybridRetriever] = None
    _reranker: Optional[Reranker] = None

    @classmethod
    def get_vector_retriever(cls) -> VectorRetriever:
        if cls._vector_retriever is None:
            cls._vector_retriever = VectorRetriever()
        return cls._vector_retriever

    @classmethod
    def get_hybrid_retriever(cls) -> HybridRetriever:
        if cls._hybrid_retriever is None:
            cls._hybrid_retriever = HybridRetriever()
        return cls._hybrid_retriever

    @classmethod
    def get_reranker(cls) -> Reranker:
        if cls._reranker is None:
            cls._reranker = Reranker()
        return cls._reranker

    @classmethod
    def create_retriever(
        cls,
        retriever_type: RetrieverType = RetrieverType.VECTOR
    ):
        if retriever_type == RetrieverType.HYBRID:
            return cls.get_hybrid_retriever()
        return cls.get_vector_retriever()

    @classmethod
    def search(
        cls,
        query: str,
        retriever_type: RetrieverType = RetrieverType.VECTOR,
        top_k: int = TOP_K_VECTOR,
        use_reranker: bool = ENABLE_RERANKER,
        top_k_initial: int = TOP_K_INITIAL,
        top_k_final: int = TOP_K_FINAL
    ) -> List[SearchResult]:
        retriever = cls.create_retriever(retriever_type)

        if use_reranker:
            initial_results = retriever.search(query, top_k_initial)
            reranker = cls.get_reranker()
            rerank_results = reranker.rerank_with_initial_results(
                query, initial_results, top_k_final
            )
            return [
                SearchResult(
                    id=r.id,
                    content=r.content,
                    metadata=r.metadata,
                    score=r.rerank_score,
                    rank=r.rerank_score
                )
                for r in rerank_results
            ]

        return retriever.search(query, top_k)

    @classmethod
    def reset(cls):
        cls._vector_retriever = None
        cls._hybrid_retriever = None
        cls._reranker = None
