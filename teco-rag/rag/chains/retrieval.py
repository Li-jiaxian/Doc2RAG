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


from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from itertools import chain
from typing import Callable, Dict, Iterable, Iterator, List, Optional, TypeVar, Union

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.chains.base import BaseRetrievalChain
from rag.common.configuration import settings
from rag.common.utils import logger
from rag.connector.vectorstore.base import VectorStoreBase
from rag.module.pre_retrieval.hyde_qyery import generate_hyde
from rag.module.pre_retrieval.multi_query import generate_queries
from rag.module.utils import get_reranker

T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


class DocumentWithVSId(Document):
    """
    矢量化后的文档
    """

    id: str = ""
    score: float = 3.0


def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e


@dataclass
class RetrievalChain(BaseRetrievalChain):
    """
    检索链类，用于执行文档检索和排序操作。

    属性:
        vectorstore (Optional[VectorStoreBase]): 向量存储对象，用于执行向量检索。
        retrievers (Optional[List[BaseRetriever]]): 预定义的检索器列表。
        top_k (int): 返回的最大文档数量。
        score_threshold (Union[None, float]): 文档相似度阈值。
        multi_query (bool): 是否启用多查询模式。
        route_query (bool): 是否启用查询路由。 TODO: 未实现

    方法:
        __post_init__(): 初始化重排序模型。
        _reciprocal_rank(): 执行加权倒数排名融合。
        pre_retrieval(): 预处理查询，生成查询变体。
        retrieval(): 执行文档检索。
        post_retrieval(): 对检索结果进行后处理和重排序。
        chain(): 执行完整的检索链流程。
    """

    vectorstore: Optional[VectorStoreBase] = None
    retrievers: Optional[List[BaseRetriever]] = None
    vectorstore_top_k: int = 25  # 向量数据库召回文档数量
    rerank_top_k: int = 5  # 重排序文档数量
    score_threshold: Union[None, float] = 0.0
    multi_query: bool = False  # 默认关闭多查询
    route_query: bool = False
    hyde: bool = False

    def __post_init__(self):
        """ "
        初始化rerank模型
        """
        self.reranker = (
            get_reranker(settings.reranker.model_name_or_path, settings.reranker.type)
            if settings.reranker.model_name_or_path and settings.reranker.type
            else None
        )

    def _reciprocal_rank(
        self, doc_lists: List[List[Document]], weights: List[float] = None, k=60
    ):
        """
        执行加权倒数排名融合(Weighted Reciprocal Rank Fusion)。

        该方法对多个排序列表进行加权融合,生成一个综合排序结果。
        算法详情参见: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        参数:
            doc_lists (List[List[Document]]): 多个文档排序列表,每个列表包含唯一的文档
            weights (List[float], 可选): 每个排序列表的权重。默认为None,表示所有列表权重相等
            k (int): RRF算法中的常数参数。默认为60

        返回:
            Tuple[List[Document], List[float]]: 返回两个列表的元组:
                - 第一个列表为按加权RRF分数降序排序的文档列表, 这个会把多路召回的内容进行一个去重
                - 第二个列表为对应的RRF分数列表
        """

        # 为每个文档内容关联RRF分数,用于后续排序
        # 跨检索器的重复内容将被合并并累积得分
        rrf_score: Dict[str, float] = defaultdict(float)
        if weights is None:
            weights = [1.0 for i in range(len(doc_lists))]
        for doc_list, weight in zip(doc_lists, weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[doc.page_content] += weight / (rank + k)

        # 根据内容去重文档,并按分数排序
        all_docs = chain.from_iterable(doc_lists)
        sorted_docs = sorted(
            unique_by_key(all_docs, lambda doc: doc.page_content),
            reverse=True,
            key=lambda doc: rrf_score[doc.page_content],
        )
        sorted_scores = [rrf_score[doc.page_content] for doc in sorted_docs]
        return sorted_docs, sorted_scores

    def pre_retrieval(self, query: str):
        """
        预处理查询，可能生成多个查询变体。
        """
        if self.multi_query:
            return generate_queries(query)
        return []

    def retrieval(self, query: str) -> Dict[str, List[Document]]:
        """
        执行文档检索操作。

        该方法使用向量数据库和预定义的检索器来检索与给定查询相关的文档。

        参数:
            query (str): 用于检索文档的查询字符串。

        返回:
            Dict[str, List[Document]]: 一个字典，其中键是检索方法的标识符，值是检索到的文档列表。

        注意:
            - 如果配置了向量数据库，将首先使用它进行检索。
            - 如果配置了预定义的检索器，将依次使用它们进行检索。
            - 对于每种检索方法，结果都会以单独的键值对存储在返回的字典中。
        """
        ensemble_docs = {}
        # 使用向量数据库的召回
        if self.vectorstore:
            kwargs = {}
            documents = []
            docs = self.vectorstore.search_docs(
                query, self.vectorstore_top_k, self.score_threshold, **kwargs
            )
            for doc, score in docs:
                document_info = doc.model_dump()
                document_info["id"] = document_info["id"] or doc.metadata.get("id")
                document = DocumentWithVSId(
                    **document_info,
                    score=score,
                )
                documents.append(document)
            ensemble_docs["vectorstore_retrieval_0"] = documents

        # retrieval by using predefined retrievers
        if self.retrievers:
            for i, retriever in enumerate(self.retrievers):
                try:
                    r_docs = retriever.invoke(
                        query,
                    )
                    if len(r_docs) > 0:
                        ensemble_docs[retriever.__class__.__name__ + "_" + str(i + 1)] = r_docs
                except Exception as e:
                    msg = f"使用预定义的召回器 {retriever.__class__.__name__} 检索召回文档时出错：{e}"
                    logger.error(f"{e.__class__.__name__}: {msg}", exc_info=e)

        return ensemble_docs

    def post_retrieval(
        self,
        query: str,
        docs: Dict[str, List[Document]],
    ):
        f_documents = []
        if len(docs) == 0:
            return []

        elif len(docs) > 1:  # rank by rrf
            # 将字典中的所有文档列表提取出来进行重排序
            doc_lists = [docs[retriever_name] for retriever_name in docs]
            sorted_docs, sorted_scores = self._reciprocal_rank(doc_lists)
            f_documents.extend(
                [
                    DocumentWithVSId(
                        page_content=x.page_content,
                        metadata=x.metadata,
                        score=sorted_scores[i],
                        id=x.metadata.get("id"),
                    )
                    for i, x in enumerate(sorted_docs)
                ]
            )

        else:
            ids, documents = [], docs[list(docs.keys())[0]]
            for doc in documents:
                if doc.id not in ids:
                    ids.append(doc.id)
                    f_documents.append(doc)

        # rerank by pre-trained model
        if self.reranker and len(f_documents) > 1:
            f_documents = self.reranker.rank(query, f_documents, self.rerank_top_k)
        else:
            f_documents = [{"document": doc} for doc in f_documents]
        return f_documents

    def chain(self, query: str):
        """
        执行检索链的主要流程。

        参数:
        query (str): 用户输入的查询字符串。

        流程:
        1. 预处理查询，可能生成多个查询变体。
        2. 对原始查询进行检索。
        3. 对每个查询变体进行检索，并将结果合并到主文档集中。
        4. 对检索到的文档进行后处理（重新排序）。
        5. 打印每个处理后的文档（用于调试）。

        返回:
        List[Dict]: 包含重新排序后的文档及其相关信息的列表。
        """
        queries = self.pre_retrieval(query)
        if self.hyde:  # hyde直接转换,不需要多查询
            query = generate_hyde(query)
            print("hyde query: ", query)
        docs = self.retrieval(query)
        # 多查询处理
        for i, q in enumerate(queries):
            q_docs = self.retrieval(q)
            for r_k in q_docs:
                docs[str(i + 1) + "_" + r_k] = q_docs[r_k]
        docs = self.post_retrieval(query, docs)  # 重新排序
        for d in docs:
            print(d)
        return docs
