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
from langchain_core.prompts.prompt import PromptTemplate

from rag.connector.base import llm

# class SummeryContentOutput(BaseModel):
#     content: str = Field(description="抽取后的信息")


# Default prompt
DEFAULT_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["content"],
    template="""作为一位精通信息处理的AI助手，你的任务是根据我的关键词, 对和我相关信息进行提取,

    请遵循以下原则：
        1. 保留每个片段独特的信息要点
        2. 确保关键细节和重要信息不会丢失
        3. 只保留对我的理解关键词相关信息有用的信息

    关键词: {keywords}
    相关信息: {content}
    """,
)


KEYWORDS_PROMPT = """
1. 请你根据我的输入提取1 ~ 3个关键词, 关键词之间用逗号隔开, 
2. 删除掉纯数字的关键词
输入: {query}
关键词:
"""


def generate_query_keywords(query):
    prompt = KEYWORDS_PROMPT.format(query=query)
    response = llm.invoke(prompt)
    return response.strip()


def _generate_summery_content(keywords, content):
    # 2. 进行总结抽取, 如果
    prompt = DEFAULT_SUMMARY_PROMPT.format(
        keywords=keywords,
        content=content,
        # format_instructions=parser.get_format_instructions()
    )
    response = llm.invoke(prompt)
    return response.strip()


def generate_summery_content(query: str, docs: List[Document]):
    """
    生成精简后的内容。
    """
    # Set up a parser
    # parser = PydanticOutputParser(pydantic_object=SummeryContentOutput)
    # 1. 生成关键词
    keywords = generate_query_keywords(query)
    total_context = ""
    for i, doc in enumerate(docs, start=1):
        total_context += f"\n\n片段{i}:\n{doc.page_content}"
    summery_content = _generate_summery_content(keywords, total_context)
    return summery_content.strip()
