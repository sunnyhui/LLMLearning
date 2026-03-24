import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    llm_type: str = "deepseek"
    model_name: str = "deepseek-chat"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    
    local_model_path: Optional[str] = None
    local_model_device: str = "cuda"


DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-coder"]


@dataclass
class RetrieverConfig:
    retriever_type: str = "hybrid"
    top_k: int = 10
    use_reranker: bool = False
    top_k_initial: int = 20
    top_k_final: int = 10


@dataclass
class PromptConfig:
    system_prompt: str = """你是一个专业的小说问答助手。基于提供的上下文内容，准确、详细地回答用户的问题。
请遵循以下规则：
1. 只使用上下文中提供的信息回答问题
2. 如果上下文中没有相关信息，请明确告知用户
3. 回答要准确、完整，并引用相关章节
4. 保持回答的连贯性和可读性"""

    user_prompt_template: str = """上下文内容：
{context}

用户问题：{question}

请基于以上上下文内容回答问题："""


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    
    @classmethod
    def from_env(cls) -> 'Config':
        config = cls()
        
        if os.getenv("OPENAI_API_KEY"):
            config.llm.api_key = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("OPENAI_API_BASE"):
            config.llm.api_base = os.getenv("OPENAI_API_BASE")
        
        if os.getenv("LLM_TYPE"):
            config.llm.llm_type = os.getenv("LLM_TYPE")
        
        if os.getenv("LLM_MODEL_NAME"):
            config.llm.model_name = os.getenv("LLM_MODEL_NAME")
        
        if os.getenv("LOCAL_MODEL_PATH"):
            config.llm.local_model_path = os.getenv("LOCAL_MODEL_PATH")
        
        return config
    
    @classmethod
    def for_openai(
        cls,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        api_base: Optional[str] = None
    ) -> 'Config':
        config = cls()
        config.llm.llm_type = "openai"
        config.llm.api_key = api_key
        config.llm.model_name = model_name
        config.llm.api_base = api_base
        return config
    
    @classmethod
    def for_deepseek(
        cls,
        api_key: str,
        model_name: str = "deepseek-chat"
    ) -> 'Config':
        config = cls()
        config.llm.llm_type = "deepseek"
        config.llm.api_key = api_key
        config.llm.model_name = model_name
        config.llm.api_base = DEEPSEEK_API_BASE
        return config
    
    @classmethod
    def for_local(
        cls,
        model_path: str,
        device: str = "cuda"
    ) -> 'Config':
        config = cls()
        config.llm.llm_type = "local"
        config.llm.local_model_path = model_path
        config.llm.local_model_device = device
        return config
