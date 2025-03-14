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

from typing import List, Tuple, Dict
from langfuse.decorators import observe, langfuse_context
from langchain_core.documents import Document
from langchain_core.messages.chat import ChatMessage
from langchain.prompts.chat import ChatPromptTemplate
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from rag.common.utils import settings
from rag.connector.utils import get_vectorstore
from rag.connector.base import llm, embedding_model
from rag.chains.retrieval import RetrievalChain
from rag.chains.generate import GenerateChain

from server.knowledge import KBServiceFactory

os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""

KNOWLEDGE_BASE_NAME = "test_langfuse_trace"


kb = KBServiceFactory.get_service_by_name(KNOWLEDGE_BASE_NAME)
# init components
vector_store = get_vectorstore(KNOWLEDGE_BASE_NAME,
                               settings.vector_store.type,
                               embedding_model)
retrieval_chain = RetrievalChain(vectorstore=vector_store,
                                 score_threshold=0.)
generate_chain = GenerateChain(llm=llm, stream=False)


@observe(name="Pre Retrieval")
def pre_retrieval(question: str) -> List[str]:
    res = retrieval_chain.pre_retrieval(question)
    langfuse_context.update_current_observation(
        input=question, output=res
    )
    return res


@observe(name="Retrieval")
def retrieval(questions: List[str]) -> Dict[str, List[Document]]:
    org_question, expand_questions = questions[0], questions[1:]
    docs = retrieval_chain.retrieval(org_question)
    for i, q in enumerate(expand_questions):
        q_docs = retrieval_chain.retrieval(q)
        for r_k in q_docs:
            if r_k in docs: docs[str(i + 1) + "_" + r_k] = q_docs[r_k]

    input = {"用户输入问题": org_question, "": expand_questions} if expand_questions else {"用户输入问题": org_question}
    output = {}
    for rk in docs:
        output[rk] = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs[rk]]

    langfuse_context.update_current_observation(
        input=input,
        output=output
    )
    return docs


@observe(name="Post Retrieval")
def post_retrieval(question: str, docs: Dict[str, List[Document]]) -> List[Document]:
    res = retrieval_chain.post_retrieval(question, docs)
    langfuse_context.update_current_observation(
        input={"用户输入问题": question, "检索召回的文档集合": docs},
        output={f"重排序后的的第{i+1}个文档": {"page_content": doc["document"].page_content, "metadata": doc["document"].metadata} for i, doc in enumerate(res)}
    )
    return res


@observe(name="Retrieval Chain")
def retriever(question: str) -> List[Document]:
    expand_questions = pre_retrieval(question)
    docs = retrieval([question] + expand_questions)
    post_docs = [doc["document"] for doc in post_retrieval(question, docs)]
    langfuse_context.update_current_observation(
        input=question, output={f"检索召回的第{i+1}个文档": {"page_content": doc.page_content, "metadata": doc.metadata} for i, doc in enumerate(post_docs)}
    )
    return post_docs


@observe(name="Augment Context")
def augment_context(question: str,
                    retrieval_docs: List[Document],
                    history: List[Tuple[str, str]]):
    messages = [{"role": h[0], "content": h[1]} for h in history] + [
        {"role": "user", "content": generate_chain.augment(question, retrieval_docs)}
    ]
    langfuse_context.update_current_observation(
        input={"用户输入问题": question , "检索召回的文档":retrieval_docs},
        output=messages
    )
    return messages


@observe(name="LLM Call", as_type="generation")
def llm_call(messages):
    input_ms = messages.copy()
    messages = [ChatMessage(role=ms["role"], content=ms["content"]) for ms in messages]
    prompt = ChatPromptTemplate.from_messages(messages).format_prompt().to_string()
    res_generator = generate_chain.generate(prompt=prompt)
    res = ""
    for ans in res_generator: res = ans
    chat_completion = {"choices": [
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content=res, role='assistant', function_call=None, tool_calls=None
            )
        )
    ]}
    langfuse_context.update_current_observation(
        input=input_ms,
        output=res
    )
    return chat_completion


@observe(name="Augmented Generate")
def generator(question: str, docs: List[Document], history: List[Tuple[str, str]]):
    messages = augment_context(question, docs, history)
    chat_completion = llm_call(messages)
    langfuse_context.update_current_observation(
        input={"用户输入问题": question, "检索召回的文档": docs},
        output=chat_completion["choices"][0].message.content
    )
    return chat_completion["choices"][0].message.content


@observe(name="Teco Rag Pipeline Traceable")
def chat_pipeline(question: str):
    retrieval_docs = retriever(question)
    result = generator(question, retrieval_docs, [])
    langfuse_context.update_current_observation(
        input={"用户输入问题": question},
        output=result
    )
    return result


question = "平时怎么预防支气管炎？"

# retrieval_docs = retriever(question)
# for doc in retrieval_docs: print(doc)

res = chat_pipeline(question)
print(res)
