import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_system import QAEngine


DEEPSEEK_API_KEY = "your-deepseek-api-key-here"

TEST_QUESTIONS = [
    "修罗阵是什么？",
    "主角是谁？",
    "故事发生在什么背景下？",
]


def test_retriever_only():
    print("\n" + "=" * 60)
    print("测试 1: 仅检索功能（不需要 API Key）")
    print("=" * 60)
    
    engine = QAEngine(
        retriever_type="hybrid",
        top_k=5,
        llm_type="deepseek"
    )
    
    query = "修罗阵"
    print(f"\n检索查询: {query}")
    
    results = engine.search_only(query, top_k=5)
    
    print(f"\n找到 {len(results)} 个相关文档:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] 章节: {r['chapter']}")
        print(f"    相关度: {r['score']:.4f}")
        print(f"    内容: {r['content'][:100]}...")
        print()


def test_qa_with_deepseek():
    print("\n" + "=" * 60)
    print("测试 2: 完整问答功能（使用 DeepSeek）")
    print("=" * 60)
    
    api_key = os.getenv("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY
    
    if api_key == "your-deepseek-api-key-here":
        print("\n⚠️  请设置 DeepSeek API Key:")
        print("   方式1: 设置环境变量 DEEPSEEK_API_KEY")
        print("   方式2: 修改本文件中的 DEEPSEEK_API_KEY 变量")
        print("\n   获取 API Key: https://platform.deepseek.com/")
        return
    
    engine = QAEngine.from_deepseek(
        api_key=api_key,
        model_name="deepseek-chat",
        retriever_type="hybrid",
        top_k=5
    )
    
    for question in TEST_QUESTIONS:
        print(f"\n{'─' * 50}")
        print(f"问题: {question}")
        print('─' * 50)
        
        try:
            result = engine.ask(question)
            
            print(f"\n答案:\n{result.answer}")
            print(f"\n置信度: {result.confidence:.2f}")
            
            print("\n来源文档:")
            for i, source in enumerate(result.sources, 1):
                print(f"  [{i}] 章节: {source.get('chapter', '未知')}")
                print(f"      相关度: {source.get('score', 0):.4f}")
        
        except Exception as e:
            print(f"❌ 错误: {e}")


def test_single_query():
    print("\n" + "=" * 60)
    print("测试 3: 单次查询示例")
    print("=" * 60)
    
    api_key = os.getenv("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY
    
    if api_key == "your-deepseek-api-key-here":
        print("\n⚠️  请设置 DeepSeek API Key 后运行此测试")
        return
    
    engine = QAEngine.from_deepseek(
        api_key=api_key,
        top_k=3
    )
    
    question = "修罗阵有什么作用？"
    print(f"\n问题: {question}")
    
    result = engine.ask(question)
    print(result)


def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "RAG 问答系统测试" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    
    test_retriever_only()
    
    test_qa_with_deepseek()
    
    test_single_query()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
