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


from abc import ABC, abstractmethod
import operator
from typing import List, Tuple
from langchain_core.documents import Document


class VectorStoreBase(ABC):
    """向量存储的基类实现"""

    @abstractmethod
    def create_vectorstore(self) -> None:
        """创建向量存储"""
        pass

    @abstractmethod
    def drop_vectorstore(self) -> None:
        """删除向量存储"""
        pass

    @abstractmethod
    def clear_vectorstore(self) -> None:
        """清空向量存储"""
        pass

    @abstractmethod
    def add_doc(self, file, docs):
        """添加文档到向量存储

        Args:
            file: 文件对象或文件路径
            docs: 要添加的文档列表
        """
        pass

    @abstractmethod
    def delete_doc(self, filename):
        """从向量存储中删除文档

        Args:
            filename: 要删除的文档的文件名
        """
        pass

    @abstractmethod
    def update_doc(self, file, docs):
        """更新向量存储中的文档

        Args:
            file: 文件对象或文件路径
            docs: 要更新的文档列表

        Returns:
            dict: 包含更新后文档ID和元数据的字典，格式为 {"id": id, "metadata": doc.metadata}
        """
        pass

    @abstractmethod
    def search_docs(self, text, top_k, threshold, **kwargs) -> List[Tuple[Document, float]]:
        """在向量存储中搜索文档

        Args:
            text: 搜索查询文本
            top_k: 返回的最大结果数
            threshold: 相似度阈值
            **kwargs: 其他可选参数

        Returns:
            List[Tuple[Document, float]]: 搜索结果列表，每个元素为(文档, 相似度分数)的元组
        """
        pass


