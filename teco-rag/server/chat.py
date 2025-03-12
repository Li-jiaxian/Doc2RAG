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

import json
from typing import List, Tuple

from fastapi import Body
from sse_starlette.sse import EventSourceResponse

from rag.chains.generate import GenerateChain
from rag.chains.retrieval import RetrievalChain
from rag.common.configuration import settings
from rag.common.utils import logger
from rag.connector.base import embedding_model, llm
from rag.connector.utils import get_vectorstore
from server.knowledge import KBServiceFactory
from server.utils import BaseResponse


async def knowledge_base_chat(
    query: str = Body(..., description="用户输入", examples=["你好"]),
    knowledge_base_name: str = Body(..., description="知识库名称", examples=["rag"]),
    history: List[Tuple[str, str]] = Body(
        [],
        description="历史对话",
        examples=[[("user", "我们来玩成语接龙，我先来，生龙活虎"), ("assistant", "虎头虎脑")]],
    ),
    score_threshold: float = Body(-2, description="相似度阈值"),
    vectorstore_top_k: int = Body(40, description="向量数据库召回的相似度文档数量"),
    rerank_top_k: int = Body(5, description="重排序的相似度文档数量"),
    stream: bool = Body(True, description="流式输出"),
    return_docs: bool = Body(False, description="返回检索结果"),
):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    vector_store = get_vectorstore(
        knowledge_base_name=knowledge_base_name,
        vs_type=settings.vector_store.type,
        embed_model=embedding_model,
    )

    # 知识库召回上下文
    # keyword_retriever = KeywordRetriever(name=knowledge_base_name, k=rerank_top_k)
    retrieval_chain = RetrievalChain(
        vectorstore=vector_store,
        score_threshold=score_threshold,
        vectorstore_top_k=vectorstore_top_k,
        rerank_top_k=rerank_top_k,
        retrievers=[],
    )
    docs = retrieval_chain.chain(query=query)
    # docs = [doc["document"] for doc in docs]
    # 记录检索到的文档
    logger.info(f"Retrieved documents for query '{query}': {docs}")

    # LLM generate
    generate_chain = GenerateChain(llm=llm, stream=stream)
    # 初始化返回结果字典
    results = {}
    llm_input_docs = []
    # 如果需要返回检索文档信息
    max_score_doc = None
    max_score = float("-inf")
    doc_results = []

    for doc in docs:
        # 更新向量召回分数为rerank分数
        doc["document"].score = doc["score"]

        # 记录最高分文档
        if doc["score"] > max_score:
            max_score = doc["score"]
            max_score_doc = doc["document"]

        if doc["document"].score > score_threshold:
            llm_input_docs.append(doc["document"])

        if return_docs:
            doc_info = {
                "filename": doc["document"].metadata.get("filename"), 
                "context": doc["document"].page_content,
                "similarity_score": doc["score"],  # rerank分数
            }
            doc_results.append(doc_info)

    # 如果没有文档超过阈值，添加最高分文档
    if not llm_input_docs and max_score_doc:
        llm_input_docs.append(max_score_doc)
        logger.warning(
            f"所有文档相似度都低于阈值{score_threshold}, 仅使用最高分文档(分数:{max_score})"
        )

    if return_docs:
        results["docs"] = doc_results
    async def iterator():
        res_generator = generate_chain.chain(query=query, docs=llm_input_docs, history=history)
        for ans in res_generator:
            results["result"] = ans
            yield json.dumps(results, ensure_ascii=False)

    return EventSourceResponse(iterator())
