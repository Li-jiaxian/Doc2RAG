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

from functools import lru_cache
import importlib
from rag.common.utils import logger

from rag.module.post_retrieval.reranker import Reranker


@lru_cache()
def get_reranker(model_name_or_path: str,
                 reranker_type: str):

    logger.info(f"Loading {model_name_or_path} as model reranker")
    if reranker_type == "rank":
        reranker = Reranker(model_name_or_path)

    return reranker


from langchain_community.document_loaders import UnstructuredFileLoader


def get_loader(name):
    """根据类名获取文档加载器, 其实就是进行动态的import

    Args:
        name (str): 加载器名称,如果包含"Customized"则从rag.module.indexing.loader获取自定义加载器,
                   否则从langchain.document_loaders获取内置加载器

    Returns:
        Type[BaseLoader]: 返回加载器类,如果获取失败则返回UnstructuredFileLoader作为默认加载器
    """
    try:
        if "Customized" in name:
            customized_document_loaders_module = importlib.import_module("rag.module.indexing.loader")
            return getattr(customized_document_loaders_module, name)
        else:
            document_loaders_module = importlib.import_module("langchain_community.document_loaders")
            return getattr(document_loaders_module, name)
    except Exception as e:
        return UnstructuredFileLoader



