import chromadb
from typing import List
from chromadb.api.types import EmbeddingFunction, Documents
from .base_retriever import BaseRetriever, SearchResult
from .config import CHROMA_DB_PATH, COLLECTION_NAME, TOP_K_VECTOR

# 创建自定义嵌入函数类
class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path)
    
    def __call__(self, input: Documents) -> list:
        if isinstance(input, str):
            return self.model.encode([input]).tolist()
        return self.model.encode(input).tolist()

# 从主配置导入模型路径
import sys
sys.path.append('..')
from config import MODEL_NAME


class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        top_k: int = TOP_K_VECTOR
    ):
        super().__init__(top_k=top_k)
        self.client = chromadb.PersistentClient(path=db_path)
        # 创建与向量化时相同的嵌入函数
        embedding_func = CustomEmbeddingFunction(MODEL_NAME)
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_func
        )

    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        if top_k is None:
            top_k = self.top_k

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        search_results = []
        for rank, (chunk_id, doc, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            search_results.append(self._create_search_result(
                chunk_id=chunk_id,
                content=doc,
                metadata=metadata,
                score=distance,
                rank=rank
            ))

        return search_results

    def get_collection_info(self) -> dict:
        return {
            "name": self.collection.name,
            "count": self.collection.count()
        }
