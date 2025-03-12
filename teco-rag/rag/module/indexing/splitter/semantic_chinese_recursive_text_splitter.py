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


from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class SemanticChineseRecursiveTextSplitter:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        # self.semantic_spliter: SemanticChunker = SemanticChunker(
        #     embedding_model,
        #     sentence_split_regex=r"(?<=[。！？!?])\s*|(?:\n\n)|(?:\d+\.\d+)|(?:[0-9]+)",
        # )
        separators = [
            # r"\n\n|",
            r"[一二三四五六七八九十]+、.+"
            # r"^0\d\s{1,2}",
            # r"(?<=[。！？])\s{0,2}",
            # r"(?<=\d\.\d{1,2})",
            # r"(?<=\d{1,2})"
        ]
        self.chinese_recursive_spliter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                keep_separator=True,
                is_separator_regex=True,
                separators=separators,
                **kwargs,
            )
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档进行语义分块和中文递归分块

        Args:
            documents (List[Document]): 待分块的文档列表

        Returns:
            List[Document]: 分块后的文档列表。先进行语义分块,再对语义块进行中文递归分块
        """
        # semantic_chunks = self.semantic_spliter.split_documents(documents)
        result_chunks = self.chinese_recursive_spliter.split_documents(documents)
        # 删除空的chunk
        result_chunks = [
            chunk for chunk in result_chunks if len(chunk.page_content.strip()) > 0
        ]
        return result_chunks
