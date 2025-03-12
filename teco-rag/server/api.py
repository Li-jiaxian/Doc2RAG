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

import uvicorn
from fastapi import FastAPI

from server.chat import knowledge_base_chat
from server.knowledge import (
    clear_knowledge_base,
    create_knowledge_base,
    delete_knowledge_base,
    get_kb_details,
    list_kbs,
    upload_docs,
)
from server.rag_test import rag_test
from server.trace import trace_rag_pipeline
from server.utils import BaseResponse, ListResponse

VERSION = "v1.0"


def create_app(run_mode: str = None):
    app = FastAPI(title="Teco-RAG API Server", version=VERSION)
    mount_app_routes(app, run_mode=run_mode)
    return app


def mount_app_routes(app: FastAPI, run_mode: str = None):
    # rag
    app.post("/rag/hit_test", tags=["Rag"], summary="rag召回测试", response_model=ListResponse)(
        rag_test
    )

    # 知识库相关
    app.post(
        "/knowledge_base/get_kb_details",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="获取知识库详情",
    )(get_kb_details)
    app.post(
        "/knowledge_base/list_knowledge_bases",
        tags=["Knowledge Base Management"],
        response_model=ListResponse,
        summary="获取知识库列表",
    )(list_kbs)
    app.post(
        "/knowledge_base/create_knowledge_base",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="创建知识库",
    )(create_knowledge_base)
    app.post(
        "/knowledge_base/delete_knowledge_base",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="删除知识库",
    )(delete_knowledge_base)
    app.post(
        "/knowledge_base/clear_knowledge_base",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="清空知识库",
    )(clear_knowledge_base)
    app.post(
        "/knowledge_base/upload_docs",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="上传文件到知识库，并/或进行向量化",
    )(upload_docs)

    # 对话相关
    app.post("/chat/knowledge_base_chat", tags=["Chat"], summary="与知识库对话")(
        knowledge_base_chat
    )

    # 开发接口
    app.post("/observability/trace_rag_pipeline", tags=["Tool"], summary="监控rag流程")(
        trace_rag_pipeline
    )


def run_api(host, port, **kwargs):
    app = create_app()
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_keyfile=kwargs.get("ssl_keyfile"),
            ssl_certfile=kwargs.get("ssl_certfile"),
        )
    else:
        uvicorn.run(app, host=host, port=port)
