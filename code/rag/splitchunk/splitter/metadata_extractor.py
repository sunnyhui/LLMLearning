from typing import List, Dict, Any
from .config import MAIN_CHARACTERS, SOURCE_FILE


class MetadataExtractor:
    def __init__(self, character_list: List[str] = None):
        self.character_list = character_list or MAIN_CHARACTERS
    
    def extract(
        self,
        chunk_content: str,
        chapter_title: str,
        chapter_index: int,
        chunk_index: int
    ) -> Dict[str, Any]:
        characters = self.extract_characters(chunk_content)
        
        metadata = {
            "source": SOURCE_FILE,
            "chapter": chapter_title,
            "chapter_index": chapter_index,
            "chunk_index": chunk_index,
            "characters": characters
        }
        
        return metadata
    
    def extract_characters(self, text: str) -> List[str]:
        found_characters = []
        
        for character in self.character_list:
            if character in text:
                found_characters.append(character)
        
        return found_characters
    
    def add_custom_characters(self, characters: List[str]) -> None:
        for char in characters:
            if char not in self.character_list:
                self.character_list.append(char)
