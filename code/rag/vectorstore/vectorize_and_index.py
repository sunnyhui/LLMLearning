import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from config import (
    CHUNKS_FILE,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    BATCH_SIZE,
    MODEL_NAME
)


def load_chunks(file_path: str) -> Dict[str, Any]:
    print(f"正在加载文本块: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"加载完成，共 {data['metadata']['total_chunks']} 个文本块")
    return data


def initialize_chromadb(db_path: str, collection_name: str):
    print(f"正在初始化ChromaDB: {db_path}")
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"集合 '{collection_name}' 已存在，包含 {collection.count()} 条记录")
        print("将删除旧集合并创建新集合...")
        client.delete_collection(name=collection_name)
    except:
        print(f"创建新集合: {collection_name}")
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    return client, collection


def store_to_chromadb(collection, chunks: List[Dict[str, Any]]):
    print(f"\n正在存储到ChromaDB并构建向量索引...")
    
    ids = [chunk['id'] for chunk in chunks]
    documents = [chunk['content'] for chunk in chunks]
    
    metadatas = []
    for chunk in chunks:
        metadata = chunk['metadata'].copy()
        if 'characters' in metadata and not metadata['characters']:
            del metadata['characters']
        metadatas.append(metadata)
    
    batch_size = BATCH_SIZE
    for i in tqdm(range(0, len(chunks), batch_size), desc="存储进度"):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
    
    print(f"存储完成，集合中共有 {collection.count()} 条记录")


def main():
    print("="*60)
    print("RAG知识库向量化与索引构建")
    print("="*60)
    
    data = load_chunks(CHUNKS_FILE)
    chunks = data['chunks']
    
    client, collection = initialize_chromadb(
        CHROMA_DB_PATH, 
        COLLECTION_NAME
    )
    
    store_to_chromadb(collection, chunks)
    
    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)
    print(f"向量数据库路径: {CHROMA_DB_PATH}")
    print(f"集合名称: {COLLECTION_NAME}")
    print(f"总文本块数: {len(chunks)}")
    print("使用模型: ChromaDB默认模型 (all-MiniLM-L6-v2)")
    print("="*60)


if __name__ == "__main__":
    main()
