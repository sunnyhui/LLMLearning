import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Optional

from qa_engine import QAEngine, QAResult
from config import Config


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              小说问答系统 - RAG QA System                    ║
║                                                              ║
║  基于 LangChain + Chroma + Sentence Transformer              ║
║  使用混合检索策略 (Vector + BM25)                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_result(result: QAResult):
    print("\n" + "─" * 60)
    print(f"📝 问题: {result.question}")
    print("─" * 60)
    print(f"\n💡 答案:\n{result.answer}")
    print("\n" + "─" * 60)
    print(f"📊 置信度: {result.confidence:.2f}")
    print("─" * 60)
    print("\n📚 来源文档:")
    for i, source in enumerate(result.sources, 1):
        print(f"\n  [{i}] 章节: {source.get('chapter', '未知')}")
        print(f"      相关度: {source.get('score', 0):.4f}")
        preview = source.get('content_preview', '')
        print(f"      内容预览: {preview}")
    print("\n" + "=" * 60)


def create_engine(args) -> QAEngine:
    if args.llm_type == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ 错误: 未设置 OpenAI API Key")
            print("请通过 --api-key 参数或 OPENAI_API_KEY 环境变量设置")
            sys.exit(1)
        
        return QAEngine(
            retriever_type=args.retriever_type,
            top_k=args.top_k,
            llm_type="openai",
            model_name=args.model,
            api_key=api_key,
            api_base=args.api_base
        )
    
    elif args.llm_type == "deepseek":
        api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ 错误: 未设置 DeepSeek API Key")
            print("请通过 --api-key 参数或 DEEPSEEK_API_KEY 环境变量设置")
            print("获取 API Key: https://platform.deepseek.com/")
            sys.exit(1)
        
        return QAEngine(
            retriever_type=args.retriever_type,
            top_k=args.top_k,
            llm_type="deepseek",
            model_name=args.model,
            api_key=api_key
        )
    
    elif args.llm_type == "local":
        model_path = args.local_model_path
        if not model_path:
            print("❌ 错误: 未设置本地模型路径")
            print("请通过 --local-model-path 参数设置")
            sys.exit(1)
        
        return QAEngine(
            retriever_type=args.retriever_type,
            top_k=args.top_k,
            llm_type="local",
            local_model_path=model_path
        )
    
    else:
        print(f"❌ 不支持的 LLM 类型: {args.llm_type}")
        sys.exit(1)


def interactive_mode(engine: QAEngine):
    print("\n🎯 进入交互模式 (输入 'quit' 或 'exit' 退出)")
    print("输入 'clear' 清除对话历史")
    print("输入 'history' 查看对话历史")
    print("输入 'search <query>' 仅进行检索\n")
    
    while True:
        try:
            user_input = input("\n🤔 请输入问题: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见!")
                break
            
            if user_input.lower() == 'clear':
                engine.clear_history()
                print("✅ 对话历史已清除")
                continue
            
            if user_input.lower() == 'history':
                history = engine.get_conversation_history()
                if not history:
                    print("📭 暂无对话历史")
                else:
                    print("\n📜 对话历史:")
                    for i, msg in enumerate(history, 1):
                        role = "👤 用户" if msg['role'] == 'user' else "🤖 助手"
                        print(f"\n[{i}] {role}:")
                        print(f"    {msg['content'][:200]}...")
                continue
            
            if user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                if query:
                    print(f"\n🔍 检索: {query}")
                    results = engine.search_only(query)
                    print(f"\n找到 {len(results)} 个相关文档:\n")
                    for i, r in enumerate(results, 1):
                        print(f"[{i}] 章节: {r['chapter']}")
                        print(f"    相关度: {r['score']:.4f}")
                        print(f"    内容: {r['content'][:100]}...\n")
                continue
            
            result = engine.ask(user_input)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()


def single_query_mode(engine: QAEngine, query: str):
    result = engine.ask(query)
    print_result(result)


def main():
    parser = argparse.ArgumentParser(
        description="小说问答系统 - RAG QA System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--llm-type",
        type=str,
        default="deepseek",
        choices=["openai", "deepseek", "local"],
        help="LLM 类型: openai, deepseek 或 local (默认: deepseek)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="模型名称 (默认: deepseek-chat)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API Key (也可通过 OPENAI_API_KEY 环境变量设置)"
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI API Base URL (可选)"
    )
    
    parser.add_argument(
        "--local-model-path",
        type=str,
        default=None,
        help="本地模型路径 (使用本地模型时必需)"
    )
    
    parser.add_argument(
        "--retriever-type",
        type=str,
        default="hybrid",
        choices=["vector", "hybrid"],
        help="检索类型: vector 或 hybrid (默认: hybrid)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="检索返回的文档数量 (默认: 10)"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="单次查询的问题 (不指定则进入交互模式)"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    print("⚙️  正在初始化系统...")
    engine = create_engine(args)
    print("✅ 系统初始化完成\n")
    
    if args.query:
        single_query_mode(engine, args.query)
    else:
        interactive_mode(engine)


if __name__ == "__main__":
    main()
