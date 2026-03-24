# RAG 问答系统搭建计划

## 项目概述

基于已有的分块、向量化存储、检索策略，搭建一个完整的小说问答系统。

## 现有资源

### 已实现模块
- **分块模块**: `splitchunk/` - 文本分块处理
- **向量化存储**: `vectorstore/` - Chroma 向量数据库
- **检索策略**: `vectorstore/retriever/` - 支持向量检索、混合检索、重排序

### 检索接口（核心使用）
```python
from retriever.retriever_factory import RetrieverFactory, RetrieverType

results = RetrieverFactory.search(
    query=query,
    retriever_type=RetrieverType.HYBRID,
    top_k=10
)
```

### SearchResult 数据结构
```python
@dataclass
class SearchResult:
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int = 0
```

---

## 新建文件夹结构

```
c:\Users\liushihui\Desktop\LLM\code\rag\qa_system\
├── __init__.py
├── config.py              # 系统配置
├── llm/
│   ├── __init__.py
│   ├── base_llm.py        # LLM 基类
│   └── langchain_llm.py   # LangChain LLM 封装
├── prompts/
│   ├── __init__.py
│   └── templates.py       # Prompt 模板
├── chain/
│   ├── __init__.py
│   └── rag_chain.py       # RAG Chain 实现
├── retriever_adapter.py   # 检索器适配器（连接现有检索系统）
├── qa_engine.py           # 问答引擎主类
└── main.py                # 主入口（CLI 交互）
```

---

## 实施步骤

### 步骤 1: 创建项目结构和配置文件

**文件**: `qa_system/__init__.py`
- 导出主要类和函数

**文件**: `qa_system/config.py`
- LLM 配置（模型名称、API Key 等）
- 检索配置（top_k、检索类型等）
- Prompt 配置

### 步骤 2: 实现检索器适配器

**文件**: `qa_system/retriever_adapter.py`
- 封装现有的 `RetrieverFactory`
- 将 `SearchResult` 转换为 LangChain 兼容格式
- 提供 `get_relevant_documents(query)` 方法

### 步骤 3: 实现 LLM 封装

**文件**: `qa_system/llm/__init__.py`
- 模块初始化

**文件**: `qa_system/llm/base_llm.py`
- LLM 基类定义
- 抽象方法：`generate(prompt)`

**文件**: `qa_system/llm/langchain_llm.py`
- 继承 LangChain 的 LLM 接口
- 支持多种 LLM 后端（OpenAI、本地模型等）
- 配置化选择

### 步骤 4: 实现 Prompt 模板

**文件**: `qa_system/prompts/__init__.py`
- 模块初始化

**文件**: `qa_system/prompts/templates.py`
- RAG Prompt 模板
- 支持自定义模板
- 包含上下文、问题、回答格式

### 步骤 5: 实现 RAG Chain

**文件**: `qa_system/chain/__init__.py`
- 模块初始化

**文件**: `qa_system/chain/rag_chain.py`
- 组合检索器、LLM、Prompt
- 实现 `invoke(query)` 方法
- 返回答案和来源文档

### 步骤 6: 实现问答引擎

**文件**: `qa_system/qa_engine.py`
- 主入口类
- 初始化所有组件
- 提供 `ask(question)` 方法
- 返回结构化结果（答案、来源、置信度）

### 步骤 7: 实现主程序

**文件**: `qa_system/main.py`
- CLI 交互界面
- 支持多轮对话
- 显示答案和来源文档
- 退出命令

---

## 技术选型

### LLM 选项
1. **OpenAI API** (推荐用于生产)
   - 需要配置 API Key
   - 使用 `langchain-openai`

2. **本地模型** (推荐用于开发测试)
   - 使用 `langchain-community` + `HuggingFacePipeline`
   - 支持模型如：Qwen、ChatGLM 等

### LangChain 组件
- `langchain-core`: 核心接口
- `langchain-community`: 社区集成
- `langchain-openai`: OpenAI 集成（可选）

---

## 依赖安装

```bash
pip install langchain langchain-core langchain-community
pip install langchain-openai  # 如果使用 OpenAI
pip install transformers torch  # 如果使用本地模型
```

---

## 使用示例

```python
from qa_system import QAEngine

# 初始化问答引擎
engine = QAEngine(
    retriever_type="hybrid",
    top_k=10,
    llm_type="local"  # 或 "openai"
)

# 提问
result = engine.ask("修罗阵是什么？")

# 获取结果
print(f"答案: {result.answer}")
print(f"来源: {result.sources}")
print(f"置信度: {result.confidence}")
```

---

## 不修改现有代码的保证

1. **检索系统**: 通过 `retriever_adapter.py` 适配现有接口，不修改 `vectorstore/retriever/` 下的任何文件
2. **配置独立**: 新系统有独立的配置文件，不依赖现有配置
3. **模块解耦**: 新系统作为独立模块，通过导入方式使用现有检索功能

---

## 文件清单

| 序号 | 文件路径 | 说明 |
|------|----------|------|
| 1 | `qa_system/__init__.py` | 模块初始化 |
| 2 | `qa_system/config.py` | 系统配置 |
| 3 | `qa_system/retriever_adapter.py` | 检索器适配器 |
| 4 | `qa_system/llm/__init__.py` | LLM 模块初始化 |
| 5 | `qa_system/llm/base_llm.py` | LLM 基类 |
| 6 | `qa_system/llm/langchain_llm.py` | LangChain LLM 封装 |
| 7 | `qa_system/prompts/__init__.py` | Prompt 模块初始化 |
| 8 | `qa_system/prompts/templates.py` | Prompt 模板 |
| 9 | `qa_system/chain/__init__.py` | Chain 模块初始化 |
| 10 | `qa_system/chain/rag_chain.py` | RAG Chain |
| 11 | `qa_system/qa_engine.py` | 问答引擎 |
| 12 | `qa_system/main.py` | 主程序入口 |

---

## 执行顺序

1. 创建文件夹结构
2. 编写配置文件
3. 实现检索器适配器
4. 实现 LLM 封装
5. 实现 Prompt 模板
6. 实现 RAG Chain
7. 实现问答引擎
8. 实现主程序
9. 测试验证
