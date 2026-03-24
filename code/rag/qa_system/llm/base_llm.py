from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def generate(
        self, 
        prompts: List[str], 
        **kwargs
    ) -> List[str]:
        pass
    
    def __call__(self, prompt: str) -> str:
        return self.invoke(prompt)
