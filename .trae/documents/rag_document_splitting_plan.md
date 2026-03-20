# RAG知识库文档分割策略实施计划

## 一、项目概述

将小说《且试天下》（content.txt，约784KB）构建为RAG知识库，采用分层分割策略确保小说叙事的连贯性和语义完整性。

## 二、文件分析结果

### 2.1 文件基本信息
- **文件路径**: `c:\Users\liushihui\Desktop\LLM\code\rag\testset\qieshitianxia\content.txt`
- **文件大小**: 784KB
- **总行数**: 约7433行
- **编码**: UTF-8

### 2.2 章节结构分析
通过正则表达式 `^[一二三四五六七八九十百千]+、` 识别章节标题，发现以下章节：

| 章节序号 | 章节标题 | 起始行号 | 备注 |
|---------|---------|---------|------|
| 引子 | 引子 | 第3行 | 开篇引言 |
| 一 | 白风夕 | 第89行 | 第一章 |
| 二 | 黑丰息 | 第455行 | 第二章 |
| 三 | 一夜宣山忽如梦 | 第759行 | 第三章 |
| 六 | 朝许夕诺可有期 | 第1903行 | 注意：跳过四、五章节 |
| 七 | 落日楼头子如玉 | 第2341行 | - |
| 八 | 借问盘中餐 | 第2795行 | - |
| 九 | 几多兵马几多悲 | 第3239行 | - |
| 十 | 断魂且了 | 第3719行 | - |
| 十一 | 春风艳舞 | 第4067行 | - |
| 十二 | 有女若东邻 | 第4459行 | - |
| 十三 | 落华纯然 | 第4899行 | - |
| 十四 | 采莲初会 | 第5269行 | - |
| 十五 | 枝头花好孰先折 | 第5649行 | - |
| 十六 | 高山流水空相念 | 第6067行 | - |
| 十七 | 归去来兮 | 第6467行 | - |
| 十八 | 风国惜云 | 第6861行 | - |
| 十九 | 白凤重现 | 第7427行 | - |

**注意**: 文件中缺少"四、五"章节标题，可能原文如此或章节命名不同，需要在实现时处理。

## 三、分层分割策略详细设计

### 3.1 第一层：章节粗分割

#### 3.1.1 章节识别规则
- **正则表达式**: `^([一二三四五六七八九十百千]+)、(.+)$` 或 `^引子$`
- **匹配模式**: 多行匹配，识别行首的章节标题
- **特殊情况处理**:
  - 引子作为特殊章节处理
  - 处理章节编号不连续的情况（如跳过四、五）

#### 3.1.2 分割逻辑
```
1. 读取完整文本
2. 使用正则表达式定位所有章节标题位置
3. 按章节标题位置切分文本
4. 每个章节作为独立文档
5. 记录章节边界信息
```

### 3.2 第二层：章节内细分割

#### 3.2.1 使用 RecursiveCharacterTextSplitter
选择 LangChain 内置的递归分割器，原因：
- 自动尝试多种分隔符
- 保持语义完整性
- 适合中文文本处理

#### 3.2.2 核心参数配置

| 参数 | 推荐值 | 说明 |
|-----|-------|------|
| `chunk_size` | 500字符 | 中文文本建议400-600字符，平衡检索精度与上下文完整性 |
| `chunk_overlap` | 80字符 | 设为chunk_size的16%，约60-100字符，避免割裂对话和场景 |
| `separators` | `["\n\n", "\n", "。", "？", "！", "；", "，", ""]` | 优先按段落和句子边界分割 |
| `length_function` | `len` | 使用字符长度计算 |

#### 3.2.3 分隔符优先级说明
1. `"\n\n"` - 段落分隔（最高优先级）
2. `"\n"` - 换行符
3. `"。"` - 句号
4. `"？"` - 问号
5. `"！"` - 感叹号
6. `"；"` - 分号
7. `"，"` - 逗号
8. `""` - 字符级分割（最低优先级，保底）

### 3.3 第三层：元数据添加

#### 3.3.1 元数据字段设计

| 字段名 | 类型 | 说明 | 示例 |
|-------|------|------|------|
| `source` | string | 源文件名 | "且试天下.txt" |
| `chapter` | string | 章节标题 | "一、白风夕" |
| `chapter_index` | int | 章节序号 | 1 |
| `chunk_index` | int | 章节内块序号 | 0, 1, 2... |
| `characters` | list[str] | 出现的主要人物 | ["白风夕", "燕瀛洲"] |
| `start_line` | int | 起始行号 | 89 |
| `end_line` | int | 结束行号 | 120 |

#### 3.3.2 主要人物关键词列表
基于小说内容，预设以下主要人物关键词：

```python
MAIN_CHARACTERS = [
    "白风夕", "风夕", "惜云公主", "风国惜云",
    "黑丰息", "丰息", "兰息公子",
    "燕瀛洲", "烈风将军",
    "玉无缘",
    "皇朝",
    "韩朴",
    "任穿云",
    "韩玄龄",
    "公无度",
    "曾甫",
    "林印安",
    "何勋",
    "令狐琚"
]
```

#### 3.3.3 人物识别逻辑
```python
def extract_characters(text: str, character_list: list) -> list:
    """从文本中提取出现的人物"""
    found = []
    for char in character_list:
        if char in text:
            found.append(char)
    return found
```

## 四、实现架构设计

### 4.1 代码模块结构

```
code/rag/testset/qieshitianxia/
├── content.txt                    # 原始文本
├── question.md                    # 问题文件
├── splitter/                      # 分割模块（新建）
│   ├── __init__.py
│   ├── chapter_splitter.py       # 章节分割器
│   ├── text_splitter.py          # 文本细分割器
│   ├── metadata_extractor.py     # 元数据提取器
│   └── config.py                 # 配置文件
├── main.py                        # 主程序入口
└── output/                        # 输出目录（新建）
    └── chunks.json                # 分割后的文本块
```

### 4.2 核心类设计

#### 4.2.1 ChapterSplitter（章节分割器）
```python
class ChapterSplitter:
    """第一层：章节分割"""
    
    def __init__(self):
        self.chapter_pattern = r'^([一二三四五六七八九十百千]+)、(.+)$'
        self.prologue_pattern = r'^引子$'
    
    def split(self, text: str) -> List[Chapter]:
        """分割章节"""
        pass
    
    def get_chapter_boundaries(self, text: str) -> List[Tuple[int, int, str]]:
        """获取章节边界"""
        pass
```

#### 4.2.2 ChunkSplitter（文本块分割器）
```python
class ChunkSplitter:
    """第二层：文本块分割"""
    
    def __init__(self, chunk_size=500, chunk_overlap=80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", "。", "？", "！", "；", "，", ""]
    
    def split(self, text: str) -> List[str]:
        """分割文本块"""
        pass
```

#### 4.2.3 MetadataExtractor（元数据提取器）
```python
class MetadataExtractor:
    """元数据提取"""
    
    def __init__(self, character_list: List[str]):
        self.character_list = character_list
    
    def extract(self, chunk: str, chapter_info: dict) -> dict:
        """提取元数据"""
        pass
    
    def extract_characters(self, text: str) -> List[str]:
        """提取人物"""
        pass
```

### 4.3 数据流程图

```
原始文本(content.txt)
    ↓
[ChapterSplitter] 章节分割
    ↓
章节文档列表 (List[Chapter])
    ↓
[ChunkSplitter] 递归分割
    ↓
文本块列表 (List[str])
    ↓
[MetadataExtractor] 元数据提取
    ↓
带元数据的文本块 (List[Document])
    ↓
输出 (chunks.json / 向量数据库)
```

## 五、实施步骤

### 步骤1：环境准备
- [ ] 确认Python环境（Python 3.8+）
- [ ] 安装依赖：langchain, langchain-text-splitters
- [ ] 创建项目目录结构

### 步骤2：实现章节分割器
- [ ] 创建 `splitter/config.py` 配置文件
- [ ] 实现 `splitter/chapter_splitter.py`
- [ ] 编写章节识别正则表达式
- [ ] 处理引子等特殊章节
- [ ] 添加单元测试

### 步骤3：实现文本块分割器
- [ ] 实现 `splitter/text_splitter.py`
- [ ] 配置 RecursiveCharacterTextSplitter 参数
- [ ] 自定义中文分隔符优先级
- [ ] 添加单元测试

### 步骤4：实现元数据提取器
- [ ] 实现 `splitter/metadata_extractor.py`
- [ ] 定义主要人物关键词列表
- [ ] 实现人物识别逻辑
- [ ] 添加行号追踪功能

### 步骤5：实现主程序
- [ ] 创建 `main.py`
- [ ] 整合三个模块
- [ ] 实现完整处理流程
- [ ] 输出JSON格式结果

### 步骤6：测试与验证
- [ ] 测试章节分割准确性
- [ ] 验证文本块语义完整性
- [ ] 检查元数据提取正确性
- [ ] 统计分割结果（章节数、块数、平均块大小等）

## 六、预期输出格式

### 6.1 JSON输出结构
```json
{
  "metadata": {
    "source_file": "且试天下.txt",
    "total_chapters": 18,
    "total_chunks": 1500,
    "split_config": {
      "chunk_size": 500,
      "chunk_overlap": 80
    }
  },
  "chunks": [
    {
      "id": "chunk_001",
      "content": "文本内容...",
      "metadata": {
        "source": "且试天下.txt",
        "chapter": "一、白风夕",
        "chapter_index": 1,
        "chunk_index": 0,
        "characters": ["白风夕", "燕瀛洲"],
        "start_line": 89,
        "end_line": 120
      }
    }
  ]
}
```

## 七、质量保证措施

### 7.1 分割质量检查
- 检查是否有章节遗漏
- 验证文本块大小分布
- 确保对话完整性（对话不被截断）
- 检查场景描述连贯性

### 7.2 元数据质量检查
- 人物识别准确率验证
- 章节序号正确性验证
- 行号范围准确性验证

## 八、后续优化建议

1. **人物关系提取**: 可扩展为提取人物关系对
2. **场景识别**: 识别战斗、对话、描写等不同场景类型
3. **时间线提取**: 提取故事发生的时间信息
4. **情感分析**: 为每个文本块添加情感标签
5. **向量数据库集成**: 直接对接Milvus/Pinecone等向量数据库

## 九、依赖清单

```txt
langchain>=0.1.0
langchain-text-splitters>=0.0.1
python>=3.8
```

## 十、时间估算

| 任务 | 预计时间 |
|-----|---------|
| 环境准备 | 10分钟 |
| 章节分割器实现 | 30分钟 |
| 文本块分割器实现 | 20分钟 |
| 元数据提取器实现 | 30分钟 |
| 主程序整合 | 20分钟 |
| 测试验证 | 20分钟 |
| **总计** | **约2小时** |
