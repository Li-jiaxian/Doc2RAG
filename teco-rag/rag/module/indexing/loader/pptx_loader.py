import os
from typing import Iterator, List

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from rag.module.indexing.loader.utils.pptx import partition_pptx


class CustomizedPPTXLoader(BaseLoader):
    """PowerPoint文件加载器,用于将PPT文件转换为Document对象.
    
    Attributes:
        is_embedding_file_name: 是否在embedding中包含文件名
        is_prefix_type: 是否在内容前添加类型前缀
        min_content_length: 内容最小长度阈值,小于该值的内容将被忽略
    """

    is_embedding_file_name = True
    is_prefix_type = False
    # 内容阈值，小于该值的内容将被忽略
    min_content_length = 80

    def __init__(self, file_path: str) -> None:
        """初始化PPT加载器.

        Args:
            file_path: PPT文件路径
        """
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        # 使用上下文管理器打开文件
        with open(file_path, "rb") as f:
            self.elements = partition_pptx(
                file=f, 
                include_metadata=True,
                include_page_breaks=True
            )

    def _process_element_text(self, element) -> str:
        """处理PPT元素文本.
        
        Args:
            element: PPT元素对象
            
        Returns:
            str: 处理后的文本
        """
        prefix: str = f"{element.category}: " if self.is_prefix_type else ""
        
        if element.category == "Table":
            return f"{prefix}{element.metadata.text_as_html}"
        elif element.category in ("Text", "Title", "UncategorizedText", "ListItem", "NarrativeText"):
            return f"{prefix}{element.text}"
        else:
            return f"{prefix}{element.category}"

    def lazy_load(self) -> Iterator[Document]:
        """惰性加载PPT内容.
        
        Yields:
            Document: 包含PPT页面内容的文档对象
        """
        current_page = 0
        page_content = []  # 使用列表存储页面内容,避免频繁字符串拼接
        
        for element in self.elements:
            if element.category == "PageBreak":
                current_page += 1
                content = "\n".join(page_content)
                if len(content) > self.min_content_length:
                    yield Document(
                        content,
                        metadata={
                            "page_number": current_page, 
                            "file_name": self.file_name
                        }
                    )
                page_content = []
                continue
                
            text = self._process_element_text(element)
            if text:
                # Title作为第一个元素时添加特殊前缀
                if element.category == "Title" and not page_content:
                    text = f"Title: {element.text}"
                page_content.append(text)

        # 处理最后一页
        if page_content:
            content = "\n".join(page_content)
            if len(content) > self.min_content_length:
                current_page += 1
                yield Document(
                    content,
                    metadata={
                        "page_number": current_page,
                        "file_name": self.file_name
                    }
                )

    def load(self) -> List[Document]:
        """加载完整PPT文件.
        
        Returns:
            List[Document]: 包含PPT内容的文档对象列表
        """
        docs = list(self.lazy_load())
        # 移除最后一页(如果内容过少)
        if docs and len(docs[-1].page_content.strip()) < self.min_content_length * 2:
            docs.pop()
        return docs
