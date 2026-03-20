from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class SearchResult:
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int = 0


@dataclass
class BaseRetriever(ABC):
    top_k: int = 5

    @abstractmethod
    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        pass

    def _create_search_result(
        self,
        chunk_id: str,
        content: str,
        metadata: Dict[str, Any],
        score: float,
        rank: int
    ) -> SearchResult:
        return SearchResult(
            id=chunk_id,
            content=content,
            metadata=metadata,
            score=score,
            rank=rank
        )
