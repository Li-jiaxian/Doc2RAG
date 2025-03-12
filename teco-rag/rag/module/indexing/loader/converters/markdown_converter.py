import re
import time
from abc import ABC, abstractmethod
from typing import List

import pdfplumber


class BaseConverter(ABC):
    """基础转换器抽象类
    
    用于将不同格式的文件转换为Markdown格式
    """
    
    def __init__(self, file_path: str) -> None:
        """初始化转换器
        
        Args:
            file_path: 待转换文件路径
        """
        self.file_path = file_path

    @abstractmethod 
    def convert_to_markdown(self) -> str:
        """将文件转换为Markdown格式
        
        Returns:
            str: 转换后的Markdown文本
        """
        pass

    @abstractmethod
    def is_my_file(self) -> bool:
        """判断文件是否可以由该转换器处理
        
        Returns:
            bool: 是否可以处理该文件
        """
        pass


class HandbookToMarkdownConverter(BaseConverter):
    """手册类PDF转Markdown转换器
    
    用于将产品手册类PDF文件转换为Markdown格式,通过字体大小和格式特征识别标题层级
    """

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        self.pdf_file = pdfplumber.open(self.file_path)
        self.total_pages = len(self.pdf_file.pages)

    def convert_to_markdown(self) -> str:
        # with open('text.md', 'r', encoding='utf-8') as f:
        #     return f.read()

        # 实现 PDF 内容到 Markdown 的转换逻辑
        start_time = time.time()
        all_text = ""
        print(f"开始转换PDF文件 {self.file_path}")
        print(f"总页数: {self.total_pages}")

        for i, page in enumerate(self.pdf_file.pages):
            print(f"正在处理第 {i+1}/{self.total_pages} 页...")
            top = page.height * 0.05
            bottom = page.height * 0.95
            page = page.crop(bbox=(0, top, page.width, bottom))
            lines = page.extract_text_lines(layout=True, y_tolerance=7)
            text = ""
            for line in lines:
                text += self.__add_title_level(line) + "\n"
            all_text += text + "\n"

        end_time = time.time()
        print(f"PDF文件 {self.file_path} 转换完成, 耗时: {end_time - start_time:.2f}秒")
        # with open("./text.md", 'w') as f:
        #     f.write(all_text)
        return all_text

    def is_my_file(self) -> bool:
        """太初的手册第一页都会有产品版本号, 发布日期"""
        if not self.file_path.endswith(".pdf"):
            return False
        page_1_content = self.pdf_file.pages[0].extract_text()
        if "产品版本号" in page_1_content and "发布日期" in page_1_content:
            return True
        return False

    def __add_title_level(self, line) -> str:
        """根据PDF文本行特征添加Markdown标题标记
        
        Args:
            line: PDF文本行信息,包含字体大小等属性
            
        Returns:
            str: 添加标题标记后的文本行
        """
        # 一级标题
        # 4. TecoDriver安装指南
        if line["chars"][0]["size"] >= 30:
            return f"# {line['text']}"
        elif line["chars"][0]["size"] >= 15:  # 二级标题和三级标题
            # 使用正则判断标题层级
            text = line["text"]
            if re.match(r"^\d+\.\d+\s+\S", text):  # 二级标题: 4.1 xxx
                return f"## {text}"
            elif re.match(r"^\d+\.\d+\.\d+\s+\S", text):  # 三级标题: 4.1.1 xxx
                return f"### {text}"
            return f"#### {text}"  # 都不符合就是四级标题
        elif line["chars"][0]["size"] >= 10:  # 五级标题
            # 可以通过正则来判断
            return f"##### {line['text']}"
        else:  # 正文
            return line["text"]

    def __del__(self):
        self.pdf_file.close()


class GovernmentPolicyPDFConverter(BaseConverter):
    """政府政策文件PDF转Markdown转换器
    
    用于转换带有政府文号的政策文件PDF,如:
    - 湘政办发〔2023〕10号
    - 合政办秘〔2022〕34号 
    """

    y_tolerance = 15

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        self.pdf_file = pdfplumber.open(self.file_path)
        self.total_pages = len(self.pdf_file.pages)

    def __is_footer(self, text: str) -> bool:
        pattern = r"^—\s*\d+\s*—$"
        return bool(re.match(pattern, text))

    def convert_to_markdown(self) -> str:
        # with open('text.md', 'r', encoding='utf-8') as f:
        # return f.read()

        # 实现 PDF 内容到 Markdown 的转换逻辑
        start_time = time.time()
        all_text = ""
        print(f"开始转换PDF文件 {self.file_path}")
        print(f"总页数: {self.total_pages}")

        for i, page in enumerate(self.pdf_file.pages):
            print(f"正在处理第 {i+1}/{self.total_pages} 页...")
            top = page.height * 0.05
            bottom = page.height * 0.88
            page = page.crop(bbox=(0, top, page.width, bottom))
            lines = page.extract_text_lines(y_tolerance=self.y_tolerance)
            text = ""
            # last_line = lines[-1]
            # if self.__is_footer(last_line["text"]):
            # lines.pop()
            for line in lines:
                # 筛选, ��除页眉页脚
                text += self.__add_title_level(line) + "\n"
            all_text += text

        end_time = time.time()
        print(f"PDF文件 {self.file_path} 转换完成, 耗时: {end_time - start_time:.2f}秒")
        # with open("./text.md", "w") as f:
        #     f.write(all_text)
        return all_text

    def is_my_file(self) -> bool:
        """政策手册第一页都会有产品版本号, 发布日期"""
        if not self.file_path.endswith(".pdf"):
            return False
        # 单层pdf, 无法使用原生解析器, 需要使用ocr
        content = self.pdf_file.pages[0].extract_text(y_tolerance=self.y_tolerance)
        if not content:
            return False
        # 匹配文件名中的政策文号格式

        policy_number_pattern1 = r"[^\s]+[〔［\[](\d{4})[〕］\]]\s*\d+\s*号"
        policy_number_pattern2 = r"〔[２0][０0][２0][０-９0-9]〕[０-９0-9]+号"
        if re.search(policy_number_pattern1, self.file_path) or re.search(
            policy_number_pattern2, self.file_path
        ):
            return True
        return False

    def __add_title_level(self, line):
        """
        根据pdf特征转markdown, 再获取知识路径
        """

        # 一级标题
        # 4. TecoDriver安装指南
        text = line["text"]
        if 30 <= line["chars"][0]["size"]:
            return f"# {line['text']}"
        # elif 14 <= line["chars"][0]["size"]:
        if re.match(r"^[一二三四五六七八九十]+、+", text):
            return f"# {line['text']}"
        elif re.match(r"^[（(][一二三四五六七八九十]+[）)]", text):
            return f"## {line['text']}"
        else:
            return text
        # else:  # 正文
        # return text

    def __del__(self):
        self.pdf_file.close()


class TablePDFConverter(BaseConverter):
    """表格类PDF转Markdown转换器
    
    专门用于处理以表格为主的PDF文件,将表格转换为Markdown格式
    """

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)
        self.pdf_file = pdfplumber.open(self.file_path)
        self.total_pages = len(self.pdf_file.pages)

    def list_to_markdown_table(self, data: List[List[str]]) -> str:
        """将二维列表转换为Markdown表格
        
        Args:
            data: 表格数据二维列表
            
        Returns:
            str: Markdown格式的表格
        """
        # 替换换行符为HTML换行标签
        processed_data = [[cell.replace("\n", "<br>") for cell in row] for row in data]

        # 创建表头
        markdown = f"| {' | '.join(processed_data[0])} |\n"

        # 添加分隔行
        markdown += f"|{'|'.join(['---' for _ in processed_data[0]])}|\n"

        # 添加数据行
        for row in processed_data[1:]:
            markdown += f"| {' | '.join(row)} |\n"

        return markdown

    def merge_tables(self, tables: List[List[str]]) -> List[List[str]]:
        """合并表格, 如果表格之间有重复的行, 则合并"""
        cur_header = tables[0]
        merge_table = [cur_header]
        for table in tables:
            for row in table[1:]:
                if row[0]:
                    merge_table.append(row)  # type: ignore
                else:
                    merge_table[-1][3] += row[3]
        return merge_table

    def convert_to_markdown(self) -> str:
        md_text = ""
        all_tables = []
        for page in self.pdf_file.pages:
            tables = page.extract_tables()
            all_tables.extend(tables)
        # merge_tables = self.merge_tables(all_tables)
        # for table in merge_tables:
        #     md_text += self.list_to_markdown_table(table) + "\n"
        for table in all_tables:
            header = table[0]
            for row in table[1:]:
                for i, cell in enumerate(row[1:]):
                    if not row[0]:
                        # 合并到上一个
                        md_text = md_text[:-3] + row[-1] + "\n"
                        break
                    else:
                        cell = cell.replace("\n", " ")
                        md_text += f"{header[i+1]}: {cell}\n"
                md_text += "\n\n"

        return md_text

    def is_my_file(self) -> bool:
        if not self.file_path.endswith(".pdf"):
            return False
        page_number = min(2, self.total_pages)
        content = ""
        for i in range(page_number):
            content += self.pdf_file.pages[i].extract_text()
        if "明细表" in content:
            return True
        return False


from typing import List

from langchain_core.documents import Document


class MarkdownPost:
    def __init__(self, md_content: str) -> None:
        self.md_content = md_content

    def add_knowledge_path(self) -> List[Document]:
        """给markdown内容添加知识路径, 根据知识路径进行加载内容"""
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=True
        )
        docs = splitter.split_text(self.md_content)

        for doc in docs:
            knowledge_path = ""
            for key, val in doc.metadata.items():
                knowledge_path += f"{val}-"
            knowledge_path = knowledge_path[:-1]
            doc.metadata["knowledge_path"] = knowledge_path
        return docs

