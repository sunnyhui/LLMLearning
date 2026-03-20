from typing import List

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
SEPARATORS = ["\n\n", "\n", "。", "？", "！", "；", "，", ""]

CHAPTER_PATTERN = r'^([一二三四五六七八九十百千]+)、(.+)$'
PROLOGUE_PATTERN = r'^引子$'

SOURCE_FILE = "且试天下.txt"

MAIN_CHARACTERS: List[str] = [
    "白风夕", "风夕", "惜云公主", "风国惜云",
    "黑丰息", "丰息", "兰息公子",
    "燕瀛洲", "烈风将军",
    "玉无缘",
    "皇朝",
    "韩朴",
    "任穿云",
    "韩玄龄",
    "公无度",
    "曾甫",
    "林印安",
    "何勋",
    "令狐琚"
]
