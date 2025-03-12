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
from langchain_core.documents import Document

def merge_small_chunks(chunks: List[Document], min_chunk_size: int = 80) -> List[Document]:
    """合并过小的文档块

    如果一个chunk内容太少(默认小于80字符),则将其与后续chunk合并,直到达到合适大小。
    这样可以避免过小的chunk缺乏足够语义信息。但是因为后面会添加知识路径和文件名进行embedding,所以可能会召回没有语义的小chunk
    Args:
        chunks: 待处理的文档块列表
        min_chunk_size: 最小chunk大小,默认80字符

    Returns:
        List[Document]: 合并后的文档块列表
    """
    merged_chunks = []
    i = 0

    while i < len(chunks):
        current_chunk = chunks[i]

        # 如果当前chunk太小且不是最后一个chunk
        if len(current_chunk.page_content) < min_chunk_size:
            # 合并当前chunk内容到下一个chunk，直到遇到一个足够大的chunk
            while i < len(chunks) - 1 and len(current_chunk.page_content) < min_chunk_size * 2:
                next_chunk = chunks[i + 1]
                current_chunk.page_content += "\n" + next_chunk.page_content
                i += 1
            merged_chunks.append(current_chunk)
        else:
            merged_chunks.append(current_chunk)
        i += 1

    # 处理最后一个chunk，如果它太小则合并到前一个
    if len(merged_chunks) > 1 and len(merged_chunks[-1].page_content) < min_chunk_size:
        last_chunk = merged_chunks[-1]
        merged_chunks[-2].page_content += "\n" + last_chunk.page_content
        merged_chunks.pop()

    return merged_chunks
