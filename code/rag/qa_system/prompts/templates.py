from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


DEFAULT_SYSTEM_PROMPT = """你是一个专业的小说问答助手。基于提供的上下文内容，准确、详细地回答用户的问题。

请遵循以下规则：
1. 只使用上下文中提供的信息回答问题
2. 如果上下文中没有相关信息，请明确告知用户
3. 回答要准确、完整，并引用相关章节
4. 保持回答的连贯性和可读性
5. 如果问题涉及人物关系、情节发展等，请根据上下文详细说明"""


DEFAULT_USER_TEMPLATE = """上下文内容：
{context}

用户问题：{question}

请基于以上上下文内容回答问题："""


class RAGPromptTemplate:
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None
    ):
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.user_template = user_template or DEFAULT_USER_TEMPLATE
        self._chat_prompt = self._build_chat_prompt()
    
    def _build_chat_prompt(self) -> ChatPromptTemplate:
        messages = [
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template(self.user_template)
        ]
        return ChatPromptTemplate.from_messages(messages)
    
    @property
    def chat_prompt(self) -> ChatPromptTemplate:
        return self._chat_prompt
    
    def format(
        self, 
        context: str, 
        question: str
    ) -> str:
        return self._chat_prompt.format(
            context=context,
            question=question
        )
    
    def format_messages(
        self, 
        context: str, 
        question: str
    ) -> List:
        return self._chat_prompt.format_messages(
            context=context,
            question=question
        )
    
    def format_context(self, documents: List) -> str:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            chapter = metadata.get('chapter', '未知章节')
            context_parts.append(f"【文档 {i}】章节: {chapter}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def format_with_documents(
        self, 
        documents: List, 
        question: str
    ) -> str:
        context = self.format_context(documents)
        return self.format(context=context, question=question)


def get_default_prompt_template() -> RAGPromptTemplate:
    return RAGPromptTemplate()


def get_custom_prompt_template(
    system_prompt: str,
    user_template: str
) -> RAGPromptTemplate:
    return RAGPromptTemplate(
        system_prompt=system_prompt,
        user_template=user_template
    )
