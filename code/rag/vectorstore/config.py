import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(BASE_DIR, "splitchunk", "output", "chunks.json")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "qieshitianxia_knowledge_base"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32
VECTOR_DIMENSION = 768

EMBEDDING_FUNCTION_NAME = "text2vec-chinese"

MODEL_CACHE_DIR = None
