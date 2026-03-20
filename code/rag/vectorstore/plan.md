# RAG知识库向量化与索引构建计划

## 项目概述
将已切分的文本块（chunks.json）转换为向量并存储到ChromaDB向量数据库中，构建RAG知识库。

## 数据分析
- **源文件**: `c:\Users\liushihui\Desktop\LLM\code\rag\splitchunk\output\chunks.json`
- **文本块数量**: 654个
- **数据格式**: JSON格式，包含metadata和chunks数组
- **每个chunk结构**:
  - id: chunk编号
  - content: 文本内容
  - metadata: 包含source、chapter、chapter_index、chunk_index等元数据

## 技术栈
- **向量化模型**: text2vec-large-chinese (通过sentence-transformers)
- **向量数据库**: ChromaDB
- **编程语言**: Python 3.10+

## 实施步骤

### 步骤1: 环境准备与依赖安装
**任务**: 创建requirements.txt并安装所需依赖

**所需依赖**:
- sentence-transformers: 用于加载text2vec-large-chinese模型
- chromadb: 向量数据库
- numpy: 数值计算
- tqdm: 进度条显示

**输出**: requirements.txt文件

### 步骤2: 创建向量化和索引构建脚本
**任务**: 创建主程序实现向量化和索引构建

**核心功能**:
1. 加载chunks.json数据
2. 初始化text2vec-large-chinese模型
3. 批量向量化文本块（考虑内存优化）
4. 初始化ChromaDB客户端
5. 创建collection并存储向量
6. 添加元数据到向量数据库
7. 构建索引

**输出**: `vectorize_and_index.py`

**关键设计决策**:
- **批处理大小**: 32-64个文本块/批次（避免内存溢出）
- **向量维度**: text2vec-large-chinese输出1024维向量
- **持久化存储**: 使用ChromaDB的持久化功能，保存到本地目录
- **元数据存储**: 将chapter、chunk_index等信息作为metadata存储

### 步骤3: 创建配置文件
**任务**: 创建配置文件管理参数

**配置项**:
- 模型名称/路径
- 向量数据库路径
- 批处理大小
- 其他超参数

**输出**: `config.py`

### 步骤4: 测试验证
**任务**: 验证向量化和索引构建结果

**验证内容**:
1. 检查向量维度是否正确（1024维）
2. 验证所有654个文本块都已向量化
3. 测试相似度搜索功能
4. 检查元数据是否正确存储

**输出**: 测试脚本和验证报告

## 文件结构
```
c:\Users\liushihui\Desktop\LLM\code\rag\
├── splitchunk/
│   └── output/
│       └── chunks.json (已存在)
├── vectorstore/
│   ├── config.py (新建)
│   ├── vectorize_and_index.py (新建)
│   ├── requirements.txt (新建)
│   └── test_search.py (新建)
└── chroma_db/ (向量数据库持久化目录，自动生成)
```

## 技术要点

### 1. text2vec-large-chinese模型使用
- 模型来源: HuggingFace (shibing624/text2vec-base-chinese 或类似)
- 输出维度: 1024维
- 适用场景: 中文文本语义相似度计算

### 2. ChromaDB使用
- 持久化存储: 使用`PersistentClient`
- Collection创建: 指定向量维度和距离度量（余弦相似度）
- 批量插入: 使用`add`方法批量添加向量和文档

### 3. 性能优化
- 批处理: 避免逐个向量化，提高效率
- 内存管理: 分批加载和处理
- 进度显示: 使用tqdm显示处理进度

## 预期结果
1. 成功将654个文本块转换为1024维向量
2. 向量和元数据存储在ChromaDB中
3. 支持基于语义相似度的检索
4. 向量数据库持久化到本地，可重复使用

## 风险与注意事项
1. **模型下载**: 首次运行需要下载模型（约1-2GB），需确保网络畅通
2. **内存占用**: text2vec-large-chinese模型和654个向量化过程需要足够内存
3. **处理时间**: 654个文本块向量化预计需要几分钟（取决于GPU/CPU）
4. **编码问题**: 确保所有文本使用UTF-8编码

## 后续扩展
- 添加增量更新功能
- 实现混合检索（向量+关键词）
- 优化检索性能
- 添加重排序机制
