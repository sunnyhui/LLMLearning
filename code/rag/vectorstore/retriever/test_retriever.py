import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retriever.vector_retriever import VectorRetriever
from retriever.hybrid_retriever import HybridRetriever
from retriever.retriever_factory import RetrieverFactory, RetrieverType
from retriever.base_retriever import SearchResult


def print_results(results, title="搜索结果"):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    if not results:
        print("未找到相关结果")
        return

    for i, result in enumerate(results, 1):
        print(f"\n【结果 {i}】")
        print(f"  ID: {result.id}")
        print(f"  章节: {result.metadata.get('chapter', 'N/A')}")
        print(f"  分数: {result.score:.4f}")
        content_preview = result.content[:150].replace('\n', ' ')
        print(f"  内容预览: {content_preview}...")
        print("-" * 70)


def test_vector_retriever():
    print("\n" + "#" * 70)
    print("#  测试 1: 基础向量检索 (Vector Retrieval)")
    print("#" * 70)

    retriever = VectorRetriever()
    info = retriever.get_collection_info()
    print(f"集合信息: {info}")

    queries = [
        "修罗阵",
    ]

    for query in queries:
        results = retriever.search(query, top_k=3)
        print_results(results, f"向量检索: '{query}'")


def test_hybrid_retriever():
    print("\n" + "#" * 70)
    print("#  测试 2: 混合检索 (Hybrid Retrieval)")
    print("#" * 70)

    try:
        retriever = HybridRetriever()

        queries = [
            "苍茫山",
            "星辰"
        ]

        for query in queries:
            results = retriever.search(query, top_k=3)
            print_results(results, f"混合检索: '{query}'")
    except ImportError as e:
        print(f"混合检索测试跳过: {e}")


def test_retriever_factory():
    print("\n" + "#" * 70)
    print("#  测试 3: 检索工厂 (Retriever Factory)")
    print("#" * 70)

    queries = ["乱世"]

    print("\n--- 切换到向量检索模式 ---")
    results = RetrieverFactory.search(
        query=queries[0],
        retriever_type=RetrieverType.VECTOR,
        top_k=3
    )
    print_results(results, f"工厂-向量检索: '{queries[0]}'")


def main():
    print("=" * 70)
    print("  RAG 检索系统测试")
    print("=" * 70)

    test_vector_retriever()

    test_hybrid_retriever()

    test_retriever_factory()

    print("\n" + "=" * 70)
    print("  测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
