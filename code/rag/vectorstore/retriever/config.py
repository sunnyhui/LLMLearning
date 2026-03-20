import os

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
CHUNKS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "splitchunk", "output", "chunks.json")
COLLECTION_NAME = "qieshitianxia_knowledge_base"

TOP_K_VECTOR = 5

ENABLE_HYBRID_SEARCH = False
BM25_ALPHA = 0.5

ENABLE_RERANKER = False
TOP_K_INITIAL = 10
TOP_K_FINAL = 5
RERANKER_MODEL = "BAAI/bge-reranker-base"
