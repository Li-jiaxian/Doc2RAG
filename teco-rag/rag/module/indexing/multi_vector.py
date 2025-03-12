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

import uuid
from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.common.utils import logger
from rag.connector.base import llm


def split_smaller_chunks(documents: List[Document], smaller_chunk_size: int):
    """将文档切分成更小的chunk

    Args:
        documents (List[Document]): 待切分的文档列表
        smaller_chunk_size (int): 更小的chunk大小

    Returns:
        List[Document]: 切分后的文档列表，每个子文档包含指向父文档的引用
    """
    # 获取所有原始文档的ID
    doc_ids = [doc.metadata["id"] for doc in documents]
    tot_docs = []

    # 创建一个新的分词器用于二次切分
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=smaller_chunk_size, chunk_overlap=0
    )

    # 遍历每个文档进行切分
    for i, doc in enumerate(documents):
        parent_id = doc_ids[i]
        # 将文档切分成更小的子文档
        sub_docs = child_splitter.split_documents([doc])

        # 为每个子文档添加元数据
        for sub_doc in sub_docs:
            sub_doc.metadata["parent_id"] = parent_id  # 添加父文档ID
            sub_doc.metadata["id"] = str(uuid.uuid4())  # 生成新的唯一ID
            sub_doc.metadata["multi_vector_type"] = (
                "text small-to-big"  # 标记为small-to-big类型
            )
            tot_docs.append(sub_doc)

    return tot_docs


TEXT_SUMMARY_TEMPLATE = """你是一位阅读能手，善于对总结归纳文章段落的摘要。
这些摘要将会被向量化并用于检索召回原始的文章段落，现在请你概括出下面这段话的要点。
文章段落内容: {text} 
总结: """


def generate_text_summaries(documents: List[Document]):
    doc_ids = [doc.metadata["id"] for doc in documents]
    tot_docs = []
    for i, doc in enumerate(documents):
        prompt = PromptTemplate.from_template(TEXT_SUMMARY_TEMPLATE).format(
            text=doc.page_content
        )
        parent_id = doc_ids[i]
        summary_doc = Document(llm.invoke(prompt))
        summary_doc.metadata["id"] = str(uuid.uuid4())
        summary_doc.metadata["parent_id"] = parent_id
        summary_doc.metadata["multi_vector_type"] = "text summary"
        tot_docs.append(summary_doc)

    return tot_docs


TABLE_SUMMARY_TEMPLATE = """你是一位阅读能手，善于对总结归纳文章中表格信息的摘要内容。
这些摘要将会被向量化并用于检索召回原始的文章段落，现在请你概括出下面这张表格的要点。
表格内容: {table} 
总结: """


def generate_table_summaries(documents: List[Document]):
    tot_docs = []
    for i, doc in enumerate(documents):
        prompt = PromptTemplate.from_template(TABLE_SUMMARY_TEMPLATE).format(
            table=doc.page_content
        )
        summary_doc = Document(llm.invoke(prompt))
        summary_doc.metadata["id"] = str(uuid.uuid4())
        summary_doc.metadata["multi_vector_type"] = "table summary"
        tot_docs.append(summary_doc)

    return tot_docs


CONTEXTUAL_TEMPLATE = """
你的任务是提供一个简短的上下文，以在整个文档中chunk，以便改进该chunk在rag的时候能够rag到
<document>
{total_text}
</document>
这是我们想要在整个文档中搜索的块
<chunk>
{chunk}
</chunk>
请提供一个简短的上下文，以在整个文档中检索到该chunk，
仅用简洁的上下文回答不超过50字，而不用其他任何东西。
"""


def generate_contextual(chunks: List[Document]):
    tot_docs = []
    for i in range(len(chunks)):
        # 获取当前chunk
        current_chunk = chunks[i]

        # 获取前后两个chunk
        start_index = max(0, i - 2)
        end_index = min(len(chunks), i + 3)
        nearby_chunks = chunks[start_index:end_index]

        # 将附近的chunk内容合并为total_text
        total_text = "\n\n".join([chunk.page_content for chunk in nearby_chunks])

        # 生成上下文
        context_prompt = CONTEXTUAL_TEMPLATE.format(
            total_text=total_text, chunk=current_chunk.page_content
        )

        # 这里可以调用llm.invoke来处理context_prompt并生成上下文文档
        contextual = llm.invoke(context_prompt)

        # 将contextual与当前chunk的page_content合并
        logger.info(f"contextual: {contextual}")
        logger.info(f"current_chunk.page_content: {current_chunk.page_content}")
        combined_content = f"{contextual}:{current_chunk.page_content}"

        # 创建新的Document对象
        context_doc = Document(page_content=combined_content)
        context_doc.metadata = current_chunk.metadata
        tot_docs.append(context_doc)

    return tot_docs
