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

from langchain_core.prompts.prompt import PromptTemplate
from rag.connector.base import llm

# Default prompt
DEFAULT_HYDE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
    # 目标: 请你帮我重写搜索信息,用于更好地去知识库里面去搜索相关信息
    # 要求
    1. 搜索信息请尽可能的精简且有用, 不超过30个字:
    2. 注意不要回答我的问题, 只需要你重写搜索信息, 方便去更好的检索到相关的信息
    3. 给1~2个重写后的搜索信息
    4. 不要改变原始搜索信息的含义, 一步一步理解原搜索真正想要表达的含义, 以便在知识库中检索到更有用的相关信息

    # 例子
    ```
    原始搜索信息: 我的Python版本为3.6，可以下载TecoPaddle的安装包吗？
    重写后搜索信息:
    TensorPaddle支持的Python语言版本范围
    TensorPaddle的环境和资源依赖
    ```
    # 现在请你重写搜索信息:
    ```
    原始搜索信息: 
    {question}
    重写后搜索信息:
    ```
    """,
)





def generate_hyde(question: str) -> str:
    """
    hyde查询, 不是标准的hyde, 算是一种变体查询
    """
    prompt = DEFAULT_HYDE_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    return question + "\n" + response.strip() # type: ignore

