from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from ..retriever_adapter import RetrieverAdapter, RetrieverAdapterConfig
from ..prompts.templates import RAGPromptTemplate, get_default_prompt_template
from ..llm.langchain_llm import create_llm, ChatLLMWrapper
from ..config import Config


@dataclass
class RAGResult:
    question: str
    answer: str
    source_documents: List[Document] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_sources_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": doc.metadata.get("id", "unknown"),
                "chapter": doc.metadata.get("chapter", "未知章节"),
                "score": doc.metadata.get("score", 0.0),
                "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            }
            for doc in self.source_documents
        ]


class RAGChain:
    def __init__(
        self,
        config: Optional[Config] = None,
        retriever: Optional[RetrieverAdapter] = None,
        llm: Optional[BaseChatModel] = None,
        prompt_template: Optional[RAGPromptTemplate] = None
    ):
        self.config = config or Config()
        
        self.retriever = retriever or self._create_retriever()
        
        self.llm = llm or self._create_llm()
        
        self.prompt_template = prompt_template or get_default_prompt_template()
        
        self._chain = self._build_chain()
    
    def _create_retriever(self) -> RetrieverAdapter:
        retriever_config = RetrieverAdapterConfig(
            retriever_type=self.config.retriever.retriever_type,
            top_k=self.config.retriever.top_k,
            use_reranker=self.config.retriever.use_reranker,
            top_k_initial=self.config.retriever.top_k_initial,
            top_k_final=self.config.retriever.top_k_final
        )
        return RetrieverAdapter(config=retriever_config)
    
    def _create_llm(self) -> BaseChatModel:
        return create_llm(
            llm_type=self.config.llm.llm_type,
            model_name=self.config.llm.model_name,
            api_key=self.config.llm.api_key,
            api_base=self.config.llm.api_base,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            local_model_path=self.config.llm.local_model_path,
            local_model_device=self.config.llm.local_model_device
        )
    
    def _build_chain(self):
        def retrieve_documents(query: str) -> List[Document]:
            return self.retriever._get_relevant_documents(query)
        
        def format_context(documents: List[Document]) -> str:
            return self.prompt_template.format_context(documents)
        
        def generate_answer(inputs: Dict[str, Any]) -> str:
            context = inputs["context"]
            question = inputs["question"]
            
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            chat_llm = ChatLLMWrapper(
                self.llm, 
                system_prompt=self.config.prompt.system_prompt
            )
            return chat_llm.invoke(prompt)
        
        chain = (
            {
                "context": RunnableLambda(retrieve_documents) | RunnableLambda(format_context),
                "question": RunnablePassthrough()
            }
            | RunnableLambda(generate_answer)
        )
        
        return chain
    
    def invoke(self, question: str) -> RAGResult:
        source_documents = self.retriever._get_relevant_documents(question)
        
        context = self.prompt_template.format_context(source_documents)
        
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        chat_llm = ChatLLMWrapper(
            self.llm,
            system_prompt=self.config.prompt.system_prompt
        )
        answer = chat_llm.invoke(prompt)
        
        confidence = self._calculate_confidence(source_documents)
        
        return RAGResult(
            question=question,
            answer=answer,
            source_documents=source_documents,
            confidence=confidence
        )
    
    def _calculate_confidence(self, documents: List[Document]) -> float:
        if not documents:
            return 0.0
        
        scores = [
            doc.metadata.get("score", 0.0) 
            for doc in documents
        ]
        
        if not scores:
            return 0.0
        
        avg_score = sum(scores) / len(scores)
        
        confidence = min(avg_score * 0.5, 1.0)
        
        return round(confidence, 2)
    
    def ask(self, question: str) -> RAGResult:
        return self.invoke(question)
