import json
import os
from typing import List, Dict, Any
from collections import defaultdict
import chromadb
from rank_bm25 import BM25Okapi

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

from .config import (
    CHROMA_DB_PATH,
    CHUNKS_FILE,
    COLLECTION_NAME,
    TOP_K_VECTOR,
    BM25_ALPHA
)
from .base_retriever import BaseRetriever, SearchResult
from .vector_retriever import VectorRetriever


def tokenize(text: str) -> List[str]:
    if JIEBA_AVAILABLE:
        return list(jieba.cut(text))
    return text.split()


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        chunks_file: str = CHUNKS_FILE,
        top_k: int = TOP_K_VECTOR,
        alpha: float = BM25_ALPHA
    ):
        super().__init__(top_k=top_k)
        self.alpha = alpha
        self.vector_retriever = VectorRetriever(db_path, collection_name)
        self.chunks_file = chunks_file
        self.bm25 = None
        self.chunks_data = None
        self._initialize_bm25()

    def _initialize_bm25(self):
        print("正在初始化 BM25 索引...")
        if not JIEBA_AVAILABLE:
            print("警告: jieba 未安装，BM25 将使用空格分词，对中文效果较差。建议运行: pip install jieba")
        self.chunks_data = self._load_chunks()
        tokenized_corpus = [tokenize(doc['content']) for doc in self.chunks_data]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 索引初始化完成，共 {len(self.chunks_data)} 个文档")

    def _load_chunks(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.chunks_file):
            raise FileNotFoundError(f"找不到 chunks 文件: {self.chunks_file}")

        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['chunks']

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        results_with_scores = []
        for idx, score in enumerate(scores):
            if score > 0:
                chunk = self.chunks_data[idx]
                results_with_scores.append((
                    chunk['id'],
                    chunk['content'],
                    chunk['metadata'],
                    score,
                    idx
                ))

        results_with_scores.sort(key=lambda x: x[3], reverse=True)
        return [
            SearchResult(
                id=item[0],
                content=item[1],
                metadata=item[2],
                score=item[3],
                rank=rank
            )
            for rank, item in enumerate(results_with_scores[:top_k], 1)
        ]

    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        if not results:
            return results
        max_score = max(r.score for r in results)
        min_score = min(r.score for r in results)
        score_range = max_score - min_score if max_score != min_score else 1

        for r in results:
            r.score = (r.score - min_score) / score_range
        return results

    def _rrf_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        k: int = 60
    ) -> List[SearchResult]:
        scores = defaultdict(float)

        for rank, result in enumerate(vector_results):
            scores[result.id] += 1 / (k + rank + 1)

        for rank, result in enumerate(bm25_results):
            scores[result.id] += 1 / (k + rank + 1)

        fused_results = []
        for chunk_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            vector_result = next((r for r in vector_results if r.id == chunk_id), None)
            bm25_result = next((r for r in bm25_results if r.id == chunk_id), None)

            if vector_result:
                content = vector_result.content
                metadata = vector_result.metadata
            else:
                content = bm25_result.content
                metadata = bm25_result.metadata

            fused_results.append(SearchResult(
                id=chunk_id,
                content=content,
                metadata=metadata,
                score=score,
                rank=len(fused_results) + 1
            ))

        return fused_results

    def _weighted_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        alpha: float = 0.5
    ) -> List[SearchResult]:
        vector_results = self._normalize_scores(vector_results)
        bm25_results = self._normalize_scores(bm25_results)

        scores = {}
        result_map = {}

        for result in vector_results:
            scores[result.id] = alpha * result.score
            result_map[result.id] = result

        for result in bm25_results:
            if result.id in scores:
                scores[result.id] += (1 - alpha) * result.score
            else:
                scores[result.id] = (1 - alpha) * result.score
                result_map[result.id] = result

        fused_results = []
        for chunk_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            result = result_map[chunk_id]
            fused_results.append(SearchResult(
                id=chunk_id,
                content=result.content,
                metadata=result.metadata,
                score=score,
                rank=len(fused_results) + 1
            ))

        return fused_results

    def search(self, query: str, top_k: int = None, alpha: float = None, fusion_method: str = "rrf") -> List[SearchResult]:
        if top_k is None:
            top_k = self.top_k
        if alpha is None:
            alpha = self.alpha

        vector_results = self.vector_retriever.search(query, top_k * 2)

        bm25_results = self._bm25_search(query, top_k * 2)

        if not vector_results and not bm25_results:
            return []

        if not vector_results:
            return bm25_results[:top_k]
        if not bm25_results:
            return vector_results[:top_k]

        if fusion_method == "weighted":
            fused_results = self._weighted_fusion(vector_results, bm25_results, alpha)
        else:
            fused_results = self._rrf_fusion(vector_results, bm25_results)
        return fused_results[:top_k]
