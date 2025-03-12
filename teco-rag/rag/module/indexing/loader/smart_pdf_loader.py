from typing import List, Optional, Type

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from rag.module.indexing.loader.converters.markdown_converter import (
    BaseConverter,
    HandbookToMarkdownConverter,
    MarkdownPost,
    GovernmentPolicyPDFConverter,
    TablePDFConverter,
)


class CustomizedSmartLoader(BaseLoader):
    """智能加载器，根据文件类型调用不同的加载器, 针对teco-rag比赛场景。
    
    支持多种文档格式的智能转换,包括:
    - 手册文档 (Handbook)
    - 政府政策文档 (Government Policy)
    - 表格文档 (Table)
    - 其他PDF文档
    """

    all_converter_classes: List[Type[BaseConverter]] = [
        HandbookToMarkdownConverter, 
        GovernmentPolicyPDFConverter, 
        TablePDFConverter
    ]

    def __init__(self, file_path: str):
        """初始化加载器
        
        Args:
            file_path: 待处理文件路径
        """
        self.file_path = file_path

    def get_converter(self) -> Optional[BaseConverter]:
        """获取适用于当前文件的转换器
        
        Returns:
            BaseConverter: 匹配的转换器实例,如果没有匹配则返回None
        """
        for converter in self.all_converter_classes:
            converter_instance = converter(self.file_path)
            if converter_instance.is_my_file():
                return converter_instance
        return None

    def load(self) -> List[Document]:
        """加载文档并转换为统一的Document格式
        
        如果没有匹配的转换器,将使用默认的PDF解析器(CustomizedPDFPlumberLoader)
        
        Returns:
            List[Document]: 转换后的文档列表
        """
        converter = self.get_converter()
        if not converter:  # 默认就用我之前定义的pdf解析器
            from rag.module.indexing.loader.plumber_pdf_loader import CustomizedPDFPlumberLoader

            plumber_pdf_loader = CustomizedPDFPlumberLoader(self.file_path, extract_images=True, dedupe=True, text_kwargs={"layout": False, "y_tolerance": 7})
            return plumber_pdf_loader.load()
        # markdown统一转成有知识路径的
        print(converter.__class__.__name__)
        markdown_content = converter.convert_to_markdown()
        return self.__add_knowledge_path(markdown_content)

    def __add_knowledge_path(self, markdown_content: str) -> List[Document]:
        """给markdown内容添加知识路径
        
        Args:
            markdown_content: markdown格式的文档内容
            
        Returns:
            List[Document]: 添加知识路径后的文档列表
        """
        return MarkdownPost(markdown_content).add_knowledge_path()
