from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .config import Config
from .chain.rag_chain import RAGChain, RAGResult
from .retriever_adapter import RetrieverAdapter, RetrieverAdapterConfig


@dataclass
class QAResult:
    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    source_documents: List[Document] = field(default_factory=list)
    
    def __str__(self) -> str:
        result = f"\n问题: {self.question}\n"
        result += f"\n答案: {self.answer}\n"
        result += f"\n置信度: {self.confidence:.2f}\n"
        result += "\n来源文档:\n"
        for i, source in enumerate(self.sources, 1):
            result += f"  [{i}] 章节: {source.get('chapter', '未知')} "
            result += f"(相关度: {source.get('score', 0):.4f})\n"
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence
        }


class QAEngine:
    def __init__(
        self,
        config: Optional[Config] = None,
        retriever_type: str = "hybrid",
        top_k: int = 10,
        llm_type: str = "deepseek",
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        local_model_path: Optional[str] = None
    ):
        if config:
            self.config = config
        else:
            self.config = Config()
            self.config.retriever.retriever_type = retriever_type
            self.config.retriever.top_k = top_k
            self.config.llm.llm_type = llm_type
            self.config.llm.model_name = model_name
            
            if api_key:
                self.config.llm.api_key = api_key
            if api_base:
                self.config.llm.api_base = api_base
            if local_model_path:
                self.config.llm.local_model_path = local_model_path
        
        self._chain = RAGChain(config=self.config)
        self._conversation_history: List[Dict[str, str]] = []
    
    def ask(self, question: str) -> QAResult:
        rag_result = self._chain.invoke(question)
        
        sources = rag_result.get_sources_summary()
        
        result = QAResult(
            question=question,
            answer=rag_result.answer,
            sources=sources,
            confidence=rag_result.confidence,
            source_documents=rag_result.source_documents
        )
        
        self._conversation_history.append({
            "role": "user",
            "content": question
        })
        self._conversation_history.append({
            "role": "assistant",
            "content": rag_result.answer
        })
        
        return result
    
    def search_only(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        retriever_config = RetrieverAdapterConfig(
            retriever_type=self.config.retriever.retriever_type,
            top_k=top_k or self.config.retriever.top_k
        )
        retriever = RetrieverAdapter(config=retriever_config)
        
        results = retriever.search(query, top_k)
        
        return [
            {
                "id": r.id,
                "content": r.content,
                "chapter": r.metadata.get("chapter", "未知"),
                "score": r.score,
                "rank": r.rank
            }
            for r in results
        ]
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self._conversation_history.copy()
    
    def clear_history(self):
        self._conversation_history.clear()
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config.retriever, key):
                setattr(self.config.retriever, key, value)
            elif hasattr(self.config.llm, key):
                setattr(self.config.llm, key, value)
        
        self._chain = RAGChain(config=self.config)
    
    @classmethod
    def from_openai(
        cls,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        api_base: Optional[str] = None,
        retriever_type: str = "hybrid",
        top_k: int = 10
    ) -> 'QAEngine':
        config = Config.for_openai(
            api_key=api_key,
            model_name=model_name,
            api_base=api_base
        )
        config.retriever.retriever_type = retriever_type
        config.retriever.top_k = top_k
        return cls(config=config)
    
    @classmethod
    def from_deepseek(
        cls,
        api_key: str,
        model_name: str = "deepseek-chat",
        retriever_type: str = "hybrid",
        top_k: int = 10
    ) -> 'QAEngine':
        config = Config.for_deepseek(
            api_key=api_key,
            model_name=model_name
        )
        config.retriever.retriever_type = retriever_type
        config.retriever.top_k = top_k
        return cls(config=config)
    
    @classmethod
    def from_local_model(
        cls,
        model_path: str,
        device: str = "cuda",
        retriever_type: str = "hybrid",
        top_k: int = 10
    ) -> 'QAEngine':
        config = Config.for_local(
            model_path=model_path,
            device=device
        )
        config.retriever.retriever_type = retriever_type
        config.retriever.top_k = top_k
        return cls(config=config)
