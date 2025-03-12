# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import os
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

from rag.chains.base import BaseIndexingChain
from rag.common.utils import logger, run_in_thread_pool
from rag.connector.database.repository.knowledge_file_repository import (
    add_docs_to_db,
    add_file_to_db,
    delete_file_from_db,
)
from rag.connector.database.utils import KnowledgeFile
from rag.connector.vectorstore.base import VectorStoreBase
from rag.module.indexing.multi_vector import (
    generate_contextual,
    generate_text_summaries,
    split_smaller_chunks,
)
from rag.module.indexing.splitter import SPLITER_MAPPING
from rag.module.indexing.splitter.utils import merge_small_chunks
from rag.module.indexing.utils import save_chunks_to_file


@dataclass
class IndexingChain(BaseIndexingChain):
    """文档索引链,用于处理文档的加载、分割、存储等流程
    
    Attributes:
        vectorstore: 向量存储实例,用于存储文档向量
        chunk_size: 文档分块大小
        chunk_overlap: 文档分块重叠大小
        zh_title_enhance: 是否增强中文标题
        multi_vector_param: 多向量参数配置
        splitter: 文本分割器
        embedding_filename: 是否在文档内容中嵌入文件名
        add_context: 是否添加上下文信息
        knowledge_path_enhance: 是否增强知识路径
        is_merge_small_chunks: 是否合并小块
    """

    vectorstore: VectorStoreBase
    chunk_size: int
    chunk_overlap: int
    zh_title_enhance: bool = False
    multi_vector_param: Dict = None
    splitter: TextSplitter = None
    embedding_filename: str = True
    add_context = False
    knowledge_path_enhance = True
    is_merge_small_chunks = True
    is_save_chunks = False

    def load(self, file: KnowledgeFile, loader: None):
        """加载文件内容
        
        Args:
            file: 知识文件对象
            loader: 文档加载器,如果为None则根据文件后缀自动选择
            
        Returns:
            加载的文档列表
        """
        if loader is None:
            loader_class = file.document_loader
        else:
            loader_class = loader
        file_path = file.filename if os.path.exists(file.filename) else file.filepath
        # 这里只返回的list只有一个document
        from langchain_community.document_loaders import (
            PDFPlumberLoader,
            UnstructuredFileLoader,
        )

        # 判断loader_class是否为UnstructuredFileLoader的子类
        if issubclass(loader_class, UnstructuredFileLoader) and file.ext in ["ppt", "pptx"]:
            docs = loader_class(file_path, mode="paged").load()
        elif issubclass(loader_class, PDFPlumberLoader):
            docs = loader_class(
                file_path, extract_images=True, text_kwargs={"layout": False}
            ).load()
        else:
            docs = loader_class(file_path).load()
        return docs

    def split(self, docs: List[Document], splitter: Union[str, TextSplitter]) -> List[Document]:
        """将文档切分成小块
        
        Args:
            docs: 待切分的文档列表
            splitter: 文本分割器,可以是预定义分割器名称或TextSplitter实例
            
        Returns:
            切分后的文档块列表
            
        处理流程:
        1. 对PPT文件特殊处理,不进行分割
        2. 使用分割器对文档进行切分
        3. 为每个chunk添加唯一ID
        4. 根据multi_vector_param参数进行二次切分和摘要生成
        5. 合并过小的chunks(如果启用)
        """
        # 如果splitter是字符串，从预定义映射中获取对应的分割器类
        chunks = []
        # PPT就不分了
        file_name = docs[0].metadata.get("file_name")
        if file_name and file_name.endswith(".pptx"):
            chunks = docs
        else:
            if isinstance(splitter, str):
                splitter = SPLITER_MAPPING[splitter]
                # 使用分割器对文档进行切分
                chunks = splitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                ).split_documents(documents=docs)
            else:
                chunks = splitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                ).split_documents(documents=docs)
            # 如果切分结果为空则返回空列表
            if not chunks:
                return []

        # 为每个chunk添加唯一ID
        for chunk in chunks:
            chunk.metadata["id"] = str(uuid.uuid4())

        # 获取多向量参数
        smaller_chunk_size = self.multi_vector_param.get("smaller_chunk_size")
        summary = self.multi_vector_param.get("summary")
        multi_vector_chunks = []

        # 如果设置了更小的chunk大小，则进行二次切分
        if smaller_chunk_size is not None and int(smaller_chunk_size) > 0:
            multi_vector_chunks.extend(split_smaller_chunks(chunks, smaller_chunk_size))

        # 如果需要生成摘要，则为每个chunk生成摘要文档
        if summary:
            multi_vector_chunks.extend(generate_text_summaries(chunks))
        final_chunks = chunks + multi_vector_chunks
        if self.is_merge_small_chunks:
            final_chunks = merge_small_chunks(final_chunks)
        return final_chunks

    def __add_knowledge_path(self, chunk):
        """为chunk添加知识路径信息
        
        Args:
            chunk: 文档块
        """
        if self.knowledge_path_enhance:
            knowledge_path = chunk.metadata.get("knowledge_path", "")
            if knowledge_path:
                chunk.page_content = f"{knowledge_path}: \n{chunk.page_content}"

    def file2chunks(self, file, **kwargs) -> Tuple[bool, Tuple[KnowledgeFile, List[Document]]]:
        """将文件转换为文档块
        
        Args:
            file: 知识文件对象
            
        Returns:
            (成功标志, (文件对象, 文档块列表))
            
        处理流程:
        1. 加载文件内容
        2. 切分文档
        3. 添加知识路径(如果启用)
        4. 添加文件名(如果启用)
        5. 添加上下文(如果启用)
        6. 保存处理后的chunks用于调试
        """
        try:
            logger.info(f"加载文件 {file.filename} 开始")
            docs = self.load(file=file, loader=None)
            chunks = self.split(docs=docs, splitter=file.text_splitter)
            if self.knowledge_path_enhance:
                for chunk in chunks:
                    self.__add_knowledge_path(chunk)
            if self.embedding_filename:
                for chunk in chunks:
                    filename = os.path.splitext(file.filename)[0]
                    chunk.page_content = f"{filename}: \n{chunk.page_content}"
            if self.add_context:
                chunks = generate_contextual(chunks)
            # NOTE 保存文档内容到文件, 用于方便调试查看chunk的, 排查问题
            if self.is_save_chunks:
                save_chunks_to_file(chunks, file.filename)
            return True, (file, chunks)
        except Exception as e:
            msg = f"从文件 {file.filename} 加载文档时出错：{e}"
            logger.error(f"{e.__class__.__name__}: {msg}", exc_info=e)
            return False, (file, msg)

    def store(self, file: KnowledgeFile, chunks: List[Document]):
        """将文件和文档块存储到数据库
        
        Args:
            file: 知识文件对象 
            chunks: 文档块列表
            
        Returns:
            存储是否成功
            
        处理流程:
        1. 删除数据库中该文件的旧记录
        2. 更新向量数据库
        3. 将新的文件和文档信息添加到数据库
        """
        # step 1. 删除db中该文件相关记录, 及其相关文档
        del_status = delete_file_from_db(file)

        # step 2. 将docs更新到向量数据库，同样需要将老记录删除
        doc_infos = self.vectorstore.update_doc(file=file, docs=chunks)

        # step 3. 将更新后的信息添加到db
        add_file_status = add_file_to_db(file, docs_count=len(chunks))
        add_docs_status = add_docs_to_db(file.kb_name, file.filename, doc_infos=doc_infos)
        # add_keyword_status = add_chuncks_keyword_to_db(doc_infos, file.kb_name)
        add_db_status = add_file_status and add_docs_status

        # 如果删除和添加都成功，则返回True
        all_status = del_status and add_db_status
        return all_status

    def chain(self, files: List[Union[KnowledgeFile, Tuple[str, str], Dict]]):
        """批量处理文件的主流程
        
        Args:
            files: 待处理的文件列表
            
        Returns:
            处理失败的文件及错误信息
            
        使用多线程并行处理文件:
        1. 将文件转换为chunks
        2. 存储到数据库
        3. 收集失败的文件信息
        """
        failed_files = {}
        kwargs_list = []
        for i, file in enumerate(files):
            kwargs = {"file": file}
            kwargs_list.append(kwargs)

        for status, result in run_in_thread_pool(func=self.file2chunks, params=kwargs_list):
            if status:
                file, chunks = result
                chunks = chunks
                self.store(file, chunks)
            else:
                file, error = result
                failed_files[file.filename] = error
        return failed_files
