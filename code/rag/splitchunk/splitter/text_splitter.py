from typing import List, Tuple
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS


class ChunkSplitter:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = SEPARATORS
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SEPARATORS,
            length_function=len,
            is_separator_regex=False
        )
    
    def split(self, text: str) -> List[dict]:
        chunks_text = self.splitter.split_text(text)
        
        chunks_with_metadata = []
        for i, chunk_text in enumerate(chunks_text):
            chunk_text = chunk_text.strip()
            if not chunk_text:  # 跳过空块
                continue
                
            chunk_data = {
                "content": chunk_text,
                "metadata": {
                    "chunk_index": i,  # 块在章节内的序号
                    "char_count": len(chunk_text),
                }
            }
            chunks_with_metadata.append(chunk_data)
        
        return chunks_with_metadata