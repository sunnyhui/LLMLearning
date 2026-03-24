from typing import Optional, List, Any
import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def create_llm(
    llm_type: str = "deepseek",
    model_name: str = "deepseek-chat",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    local_model_path: Optional[str] = None,
    local_model_device: str = "cuda",
    **kwargs
) -> BaseChatModel:
    if llm_type == "openai":
        return _create_openai_llm(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif llm_type == "deepseek":
        return _create_deepseek_llm(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif llm_type == "local":
        return _create_local_llm(
            model_path=local_model_path,
            device=local_model_device,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的 LLM 类型: {llm_type}")


def _create_openai_llm(
    model_name: str,
    api_key: Optional[str],
    api_base: Optional[str],
    temperature: float,
    max_tokens: int,
    **kwargs
) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "请安装 langchain-openai: pip install langchain-openai"
        )
    
    llm_kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if api_key:
        llm_kwargs["api_key"] = api_key
    elif os.getenv("OPENAI_API_KEY"):
        llm_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError("OpenAI API Key 未设置，请通过 api_key 参数或 OPENAI_API_KEY 环境变量设置")
    
    if api_base:
        llm_kwargs["base_url"] = api_base
    elif os.getenv("OPENAI_API_BASE"):
        llm_kwargs["base_url"] = os.getenv("OPENAI_API_BASE")
    
    llm_kwargs.update(kwargs)
    
    return ChatOpenAI(**llm_kwargs)


def _create_deepseek_llm(
    model_name: str,
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
    **kwargs
) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "请安装 langchain-openai: pip install langchain-openai"
        )
    
    deepseek_api_base = "https://api.deepseek.com/v1"
    
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API Key 未设置，请通过 api_key 参数或 DEEPSEEK_API_KEY 环境变量设置")
    
    llm_kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": api_key,
        "base_url": deepseek_api_base,
    }
    
    llm_kwargs.update(kwargs)
    
    return ChatOpenAI(**llm_kwargs)


def _create_local_llm(
    model_path: str,
    device: str,
    temperature: float,
    max_tokens: int,
    **kwargs
) -> BaseChatModel:
    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
    except ImportError:
        raise ImportError(
            "请安装相关依赖: pip install langchain-community transformers torch"
        )
    
    if not model_path:
        raise ValueError("本地模型路径未设置，请通过 local_model_path 参数设置")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        **kwargs
    )
    
    return HuggingFacePipeline(pipeline=pipe)


class ChatLLMWrapper:
    def __init__(self, llm: BaseChatModel, system_prompt: Optional[str] = None):
        self.llm = llm
        self.system_prompt = system_prompt
    
    def invoke(self, prompt: str) -> str:
        messages = []
        
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        response = self.llm.invoke(messages)
        
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    
    def generate(self, prompts: List[str]) -> List[str]:
        return [self.invoke(prompt) for prompt in prompts]
    
    def __call__(self, prompt: str) -> str:
        return self.invoke(prompt)
