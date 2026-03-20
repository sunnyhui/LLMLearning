import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .config import CHAPTER_PATTERN, PROLOGUE_PATTERN


@dataclass
class Chapter:
    title: str
    content: str
    start_line: int
    end_line: int
    index: int


class ChapterSplitter:
    def __init__(self):
        self.chapter_pattern = re.compile(CHAPTER_PATTERN, re.MULTILINE)
        self.prologue_pattern = re.compile(PROLOGUE_PATTERN, re.MULTILINE)
    
    def split(self, text: str) -> List[Chapter]:
        chapters = []
        lines = text.split('\n')
        
        chapter_positions = self._find_chapter_positions(lines)
        
        if not chapter_positions:
            return chapters
        
        for i, (line_num, title, index) in enumerate(chapter_positions):
            start_line = line_num
            if i + 1 < len(chapter_positions):
                end_line = chapter_positions[i + 1][0] - 1
            else:
                end_line = len(lines)
            
            content_lines = lines[start_line:end_line]
            content = '\n'.join(content_lines).strip()
            
            chapter = Chapter(
                title=title,
                content=content,
                start_line=start_line + 1,
                end_line=end_line,
                index=index
            )
            chapters.append(chapter)
        
        return chapters
    
    def _find_chapter_positions(self, lines: List[str]) -> List[Tuple[int, str, int]]:
        positions = []
        chapter_index = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            prologue_match = self.prologue_pattern.match(line_stripped)
            if prologue_match:
                positions.append((i, "引子", 0))
                continue
            
            chapter_match = self.chapter_pattern.match(line_stripped)
            if chapter_match:
                chapter_num = chapter_match.group(1)
                chapter_title = chapter_match.group(2)
                full_title = f"{chapter_num}、{chapter_title}"
                
                index = self._chinese_num_to_int(chapter_num)
                positions.append((i, full_title, index))
        
        return positions
    
    def _chinese_num_to_int(self, chinese_num: str) -> int:
        digits = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
                  '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
        units = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
        
        if not chinese_num:
            return 0
        
        if chinese_num in digits:
            return digits[chinese_num]
        
        result = 0
        temp = 0
        unit_value = 1
        
        for char in reversed(chinese_num):
            if char in units:
                unit_value = units[char]
                if temp != 0:
                    result += temp * unit_value
                    temp = 0
                else:
                    result += unit_value
            elif char in digits:
                temp = digits[char]
        
        result += temp
        return result