import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retriever.vector_retriever import VectorRetriever
from retriever.hybrid_retriever import HybridRetriever
from retriever.retriever_factory import RetrieverFactory, RetrieverType
from retriever.base_retriever import SearchResult
from retriever.reranker import Reranker

queries = [
        "修罗阵",
    ]
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


def print_rerank_results(results, title="重排序结果"):
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
        print(f"  重排序分数: {result.rerank_score:.4f}")
        print(f"  原始排名: {result.original_rank + 1}")
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

    for query in queries:
        results = retriever.search(query, top_k=10)
        print_results(results, f"向量检索: '{query}'")


def test_hybrid_retriever():
    print("\n" + "#" * 70)
    print("#  测试 2: 混合检索 (Hybrid Retrieval)")
    print("#" * 70)

    try:
        retriever = HybridRetriever()

        for query in queries:
            results = retriever.search(query, top_k=10)
            print_results(results, f"混合检索: '{query}'")
    except ImportError as e:
        print(f"混合检索测试跳过: {e}")


def test_reranker():
    print("\n" + "#" * 70)
    print("#  测试 3: 重排序 (Reranking)")
    print("#" * 70)

    try:
        retriever = VectorRetriever()
        reranker = Reranker()

        query = queries[0]

        print(f"\n查询: '{query}'")
        print("正在获取初始检索结果...")
        initial_results = retriever.search(query, top_k=10)
        print_results(initial_results, "初始向量检索结果 (Top-10)")

        print("\n正在进行重排序...")
        rerank_results = reranker.rerank_with_initial_results(query, initial_results, top_k=5)
        print_rerank_results(rerank_results, "重排序结果 (Top-5)")

    except ImportError as e:
        print(f"重排序测试跳过: {e}")
        print("请运行: pip install sentence-transformers")
    except Exception as e:
        print(f"重排序测试失败: {e}")


def test_retriever_factory():
    print("\n" + "#" * 70)
    print("#  测试 4: 检索工厂 (Retriever Factory)")
    print("#" * 70)

    print("\n--- 切换到向量检索模式 ---")
    results = RetrieverFactory.search(
        query=queries[0],
        retriever_type=RetrieverType.HYBRID,
        top_k=10
    )
    print_results(results, f"工厂-混合检索: '{queries[0]}'")


def main():
    print("=" * 70)
    print("  RAG 检索系统测试")
    print("=" * 70)

    # test_vector_retriever()

    # test_hybrid_retriever()

    # test_reranker()

    test_retriever_factory()

    print("\n" + "=" * 70)
    print("  测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
