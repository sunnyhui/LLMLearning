import sys
import os
from typing import List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from vectorstore.retriever.retriever_factory import RetrieverFactory, RetrieverType
from vectorstore.retriever.base_retriever import SearchResult


@dataclass
class RetrieverAdapterConfig:
    retriever_type: str = "hybrid"
    top_k: int = 10
    use_reranker: bool = False
    top_k_initial: int = 20
    top_k_final: int = 10


class RetrieverAdapter(BaseRetriever):
    config: RetrieverAdapterConfig
    
    def __init__(self, config: Optional[RetrieverAdapterConfig] = None):
        super().__init__()
        self.config = config or RetrieverAdapterConfig()
        self._retriever_type_map = {
            "vector": RetrieverType.VECTOR,
            "hybrid": RetrieverType.HYBRID
        }
    
    def _get_retriever_type(self) -> RetrieverType:
        return self._retriever_type_map.get(
            self.config.retriever_type.lower(), 
            RetrieverType.HYBRID
        )
    
    def _search_result_to_document(self, result: SearchResult) -> Document:
        return Document(
            page_content=result.content,
            metadata={
                "id": result.id,
                "score": result.score,
                "rank": result.rank,
                **result.metadata
            }
        )
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager=None
    ) -> List[Document]:
        retriever_type = self._get_retriever_type()
        
        results = RetrieverFactory.search(
            query=query,
            retriever_type=retriever_type,
            top_k=self.config.top_k,
            use_reranker=self.config.use_reranker,
            top_k_initial=self.config.top_k_initial,
            top_k_final=self.config.top_k_final
        )
        
        documents = [
            self._search_result_to_document(result) 
            for result in results
        ]
        
        return documents
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        retriever_type = self._get_retriever_type()
        k = top_k or self.config.top_k
        
        return RetrieverFactory.search(
            query=query,
            retriever_type=retriever_type,
            top_k=k
        )
