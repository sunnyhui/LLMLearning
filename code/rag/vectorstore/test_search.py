import chromadb
from config import CHROMA_DB_PATH, COLLECTION_NAME


def test_search(query: str, n_results: int = 5):
    print("="*60)
    print(f"测试查询: {query}")
    print("="*60)
    
    print("\n正在连接向量数据库...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    print(f"集合中共有 {collection.count()} 条记录\n")
    
    print(f"正在搜索最相似的 {n_results} 条记录...\n")
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    print("搜索结果:")
    print("-"*60)
    for i, (id, doc, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\n结果 {i}:")
        print(f"ID: {id}")
        print(f"章节: {metadata['chapter']}")
        print(f"相似度距离: {distance:.4f}")
        print(f"内容预览: {doc[:100]}...")
        print("-"*60)
    
    return results


def verify_collection():
    print("="*60)
    print("验证向量数据库")
    print("="*60)
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    print(f"\n集合名称: {COLLECTION_NAME}")
    print(f"总记录数: {collection.count()}")
    
    print("\n随机抽取5条记录验证:")
    results = collection.get(limit=5)
    
    for i, (id, doc, metadata) in enumerate(zip(
        results['ids'],
        results['documents'],
        results['metadatas']
    ), 1):
        print(f"\n记录 {i}:")
        print(f"ID: {id}")
        print(f"章节: {metadata['chapter']}")
        print(f"内容长度: {len(doc)} 字符")
        print(f"内容预览: {doc[:80]}...")


def main():
    print("\n" + "="*60)
    print("RAG知识库测试")
    print("="*60)
    
    verify_collection()
    
    test_queries = [
        "修罗阵"
    ]
    
    for query in test_queries:
        print("\n")
        test_search(query, n_results=5)


if __name__ == "__main__":
    main()
