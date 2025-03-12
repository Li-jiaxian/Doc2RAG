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


from fastapi import Body

from rag.chains.retrieval import RetrievalChain
from rag.common.configuration import settings
from rag.common.utils import logger
from rag.connector.base import embedding_model
from rag.connector.utils import get_vectorstore
from server.knowledge import KBServiceFactory
from server.utils import BaseResponse, ListResponse


async def rag_test(
    query: str = Body(
        ..., description="用户输入", examples=["太初元碁参与了哪几个智算中心的建设"]
    ),
    knowledge_base_name: str = Body(..., description="知识库名称", examples=["test_kb"]),
    score_threshold: float = Body(0.0, description="相似度阈值"),
    vectorstore_top_k: int = Body(10, description="向量数据库召回的相似度文档数量"),
    rerank_top_k: int = Body(5, description="重排序的相似度文档数量"),
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
    retrieval_chain = RetrievalChain(
        vectorstore=vector_store,
        score_threshold=score_threshold,
        vectorstore_top_k=vectorstore_top_k,
        rerank_top_k=rerank_top_k,
    )
    # 从检索链中获取相关文档
    retrieved_results = retrieval_chain.chain(query=query)
    docs = []
    for result in retrieved_results:
        if "document" in result:
            docs.append(result["document"])
    # 记录检索到的文档
    logger.info(f"Retrieved documents for query '{query}': {docs}")

    results = [
        {
            "filename": d.metadata.get("filename"),
            "context": d.page_content,
            "similarity_score": d.score,
        }
        for d in docs
    ]

    return ListResponse(code=200, msg="成功", data=results)
