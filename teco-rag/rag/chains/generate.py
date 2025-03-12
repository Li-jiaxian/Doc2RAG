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

from dataclasses import dataclass
from typing import List, Tuple, Union

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import LLM, BaseChatModel
from langchain_core.messages.chat import ChatMessage

from rag.chains.base import BaseGenerationChain
from rag.common.utils import get_prompt_template
from rag.module.pre_generate.summery_content import generate_summery_content


@dataclass
class GenerateChain(BaseGenerationChain):
    llm: Union[LLM, BaseChatModel]
    stream: bool = False
    prompt_type: str = "rag"
    is_summary_prompt: bool = False
    keep_top_content: bool = False

    def augment(self, query: str, docs: List[Document]):
        if self.is_summary_prompt:
            context = generate_summery_content(query, docs)
        else:
            context = ""
            for i, doc in enumerate(docs, start=1):
                context += f"\n\n片段{i}:\n{doc.page_content}"
        if self.keep_top_content: # 最相关的尽量保留因为可能其中的重要信息被删除
            context = f"{docs[0].page_content}\n\n{context}"

        prompt_template = get_prompt_template(type=self.prompt_type)
        context = PromptTemplate.from_template(prompt_template).format(
            query=query, context=context
        )
        return context

    def generate(self, prompt):
        if self.stream:
            result = ""
            for res in self.llm.stream(prompt):
                result += res.content if isinstance(self.llm, BaseChatModel) else res
                yield result
        else:
            result = self.llm.invoke(prompt)
            result = result.content if isinstance(self.llm, BaseChatModel) else result
            for res in iter([result]):
                yield res



    def chain(self, query: str, docs: List[Document], history: List[Tuple[str, str]]):
        """
        生成答案
        """
        message_list = [ChatMessage(role=h[0], content=h[1]) for h in history]
        # 1. 生成问答上下文prompt
        content = self.augment(query, docs)
        # 2. 构建langchain prompt
        prompt = (
        ChatPromptTemplate.from_messages(
                    message_list + [ChatMessage(role="user", content=content)]
            )
            .format_prompt()
            .to_string()
        )
        # 3. 生成答案
        return self.generate(prompt)
