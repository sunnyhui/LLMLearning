# RAG 检索策略实施计划

## 项目概述

在已构建的 ChromaDB 向量数据库基础上，设计和实现多层次检索策略，提升 RAG 知识库的检索精度。

## 现有基础

- **向量数据库**: ChromaDB (ChromaDB 默认使用 all-MiniLM-L6-v2 模型)
- **集合名称**: `qieshitianxia_knowledge_base`
- **文本块数量**: 654 个
- **数据来源**: `且试天下.txt`
- **向量维度**: 384 维 (ChromaDB 默认模型)
- **索引目录**: `c:\Users\liushihui\Desktop\LLM\code\rag\chroma_db`

## 检索策略架构

### 层级一：基础向量检索 (Vector Search)

**实现方式**:
- 将用户问题转换为向量表示
- 在 ChromaDB 中进行余弦相似度搜索
- 返回 Top-K 个最相似的文本块

**配置参数**:
- `TOP_K_VECTOR`: 5-10 (默认 5)
- 距离度量: 余弦相似度 (ChromaDB collection 创建时指定)

**代码位置**: `retriever/vector_retriever.py`

---

### 层级二：混合检索增强 (Hybrid Search) - 可选进阶

**实现方式**:
结合向量检索和关键词检索 (BM25)，提高对具体名称、诗词、地点的检索精度

**关键词检索实现**:
- 使用 `rank_bm25` 库实现 BM25 算法
- 在文本块内容上进行关键词匹配
- 对诗词名句、地点名称等精确匹配有良好效果

**示例场景**:
> 用户问："甄嬛写的'逆风如解意'全诗是什么？"
> - 向量检索可能难以精确匹配这句诗
> - BM25 关键词检索可以精确定位包含"逆风如解意"的文本块

**融合策略**:
```
final_score = alpha * vector_similarity + (1 - alpha) * bm25_score
```
- `alpha`: 可调节参数 (0.0 ~ 1.0)，默认 0.5

**代码位置**: `retriever/hybrid_retriever.py`

---

### 层级三：重排序 (Reranking) - 可选进阶

**实现方式**:
- 初步检索出较多结果 (如 Top-10)
- 使用重排序模型对结果进行精排
- 选出最相关的 3-5 个片段提供给 LLM

**重排序模型**:
- **推荐模型**: BAAI/bge-reranker-base 或 bge-reranker-large
- **作用**: 对初步检索结果进行更精细的相关性排序

**配置参数**:
- `TOP_K_INITIAL`: 10 (初步检索数量)
- `TOP_K_FINAL`: 3-5 (重排后最终数量)

**代码位置**: `retriever/reranker.py`

---

## 文件结构

```
c:\Users\liushihui\Desktop\LLM\code\rag\
├── vectorstore/
│   ├── config.py                    # 现有配置文件
│   ├── vectorize_and_index.py       # 现有向量化脚本
│   ├── test_search.py               # 现有测试脚本
│   └── retriever/                   # 新建：检索模块
│       ├── __init__.py
│       ├── config.py                # 新建：检索配置
│       ├── base_retriever.py        # 新建：检索基类
│       ├── vector_retriever.py      # 新建：向量检索
│       ├── hybrid_retriever.py      # 新建：混合检索
│       ├── reranker.py              # 新建：重排序
│       └── retriever_factory.py      # 新建：检索工厂
├── requirements.txt                 # 更新：添加 BM25 等依赖
└── chroma_db/                        # 现有向量数据库
```

---

## 实施步骤

### 步骤 1: 创建检索配置模块

**文件**: `retriever/config.py`

**配置项**:
```python
# 向量检索配置
TOP_K_VECTOR = 5

# 混合检索配置
ENABLE_HYBRID_SEARCH = False  # 默认关闭，可按需开启
BM25_ALPHA = 0.5  # 融合权重

# 重排序配置
ENABLE_RERANKER = False  # 默认关闭，可按需开启
TOP_K_INITIAL = 10  # 初步检索数量
TOP_K_FINAL = 5  # 最终返回数量
RERANKER_MODEL = "BAAI/bge-reranker-base"
```

---

### 步骤 2: 实现基础向量检索

**文件**: `retriever/vector_retriever.py`

**核心功能**:
1. 加载 ChromaDB 连接
2. 将用户问题转换为向量
3. 执行相似度搜索
4. 返回排序后的结果

**函数签名**:
```python
def vector_search(query: str, top_k: int = 5) -> List[SearchResult]
```

---

### 步骤 3: 实现混合检索 (可选)

**文件**: `retriever/hybrid_retriever.py`

**核心功能**:
1. 向量检索获取初步结果
2. BM25 关键词检索获取结果
3. 分数融合 (RRF 或加权融合)
4. 返回混合排序结果

**函数签名**:
```python
def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5) -> List[SearchResult]
```

---

### 步骤 4: 实现重排序 (可选)

**文件**: `retriever/reranker.py`

**核心功能**:
1. 加载重排序模型 (BAAI/bge-reranker)
2. 对初步检索结果进行成对比较
3. 计算相关性分数
4. 返回精排结果

**函数签名**:
```python
def rerank(query: str, documents: List[str], top_k: int = 5) -> List[RerankResult]
```

---

### 步骤 5: 创建检索工厂

**文件**: `retriever/retriever_factory.py`

**功能**:
- 根据配置选择检索策略
- 提供统一的检索接口
- 支持策略切换 (向量检索 / 混合检索 / 重排序)

---

### 步骤 6: 更新依赖

**文件**: `requirements.txt`

**新增依赖**:
```
rank_bm25==0.2.2        # BM25 关键词检索
sentence-transformers    # 重排序模型
```

---

### 步骤 7: 测试验证

**测试场景**:

1. **基础向量检索测试**
   - 查询: "苍茫山上的棋局是什么？"
   - 验证: 返回相关文本块

2. **混合检索测试** (可选)
   - 查询: "逆风如解意"
   - 验证: 精确匹配到包含该诗句的文本块

3. **重排序测试** (可选)
   - 查询: "白袍老者和黑袍老者下棋"
   - 验证: 重排后结果更相关

---

## 核心数据结构

### SearchResult

```python
@dataclass
class SearchResult:
    id: str                    # 文本块 ID
    content: str               # 文本内容
    metadata: Dict[str, Any]   # 元数据 (chapter, chunk_index 等)
    score: float               # 相似度分数
    rank: int                  # 排名
```

### RerankResult

```python
@dataclass
class RerankResult:
    id: str                    # 文本块 ID
    content: str               # 文本内容
    metadata: Dict[str, Any]   # 元数据
    rerank_score: float        # 重排序分数
    original_rank: int        # 原始排名
```

---

## 融合策略详解

### 1. 加权融合 (Weighted Fusion)

```python
def weighted_fusion(vector_results, bm25_results, alpha=0.5):
    """
    alpha: 向量检索权重
    (1-alpha): BM25 权重
    """
    scores = {}
    for result in vector_results:
        scores[result.id] = alpha * result.score

    for result in bm25_results:
        if result.id in scores:
            scores[result.id] += (1 - alpha) * result.score
        else:
            scores[result.id] = (1 - alpha) * result.score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 2. RRF (Reciprocal Rank Fusion)

```python
def rrf_fusion(vector_results, bm25_results, k=60):
    """
    k: 调节参数，通常设为 60
    """
    scores = defaultdict(float)

    for rank, result in enumerate(vector_results):
        scores[result.id] += 1 / (k + rank + 1)

    for rank, result in enumerate(bm25_results):
        scores[result.id] += 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## 技术选型说明

### BM25 vs 纯向量检索

| 特性 | BM25 | 向量检索 |
|------|------|----------|
| 精确匹配 | ✅ 优秀 | ⚠️ 一般 |
| 语义理解 | ❌ 无 | ✅ 优秀 |
| 诗词/地名 | ✅ 优秀 | ⚠️ 一般 |
| 长句语义 | ⚠️ 一般 | ✅ 优秀 |

### 重排序模型选择

| 模型 | 参数量 | 效果 | 适用场景 |
|------|--------|------|----------|
| BAAI/bge-reranker-base | ~110M | 良好 | 通用场景 |
| BAAI/bge-reranker-large | ~280M | 优秀 | 高精度需求 |

---

## 注意事项

1. **性能考虑**: 重排序模型较大，首次加载需要下载模型
2. **内存占用**: 混合检索和重排序会占用更多内存
3. **检索速度**: 向量检索 < 混合检索 < 重排序
4. **按需开启**: 建议先使用基础向量检索，效果不佳时再开启进阶功能

---

## 后续扩展

1. 添加多向量检索支持 (每个文本块多个向量)
2. 实现查询扩展 / 查询改写
3. 添加上下文窗口 (相邻文本块一并召回)
4. 支持多知识库检索
