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
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union

import httpx
import pydantic
import requests
from pydantic import BaseModel
from rag.common.utils import logger


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListResponse(BaseResponse):
    data: List[Any] = pydantic.Field(..., description="List of names")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ApiRequest:
    """
    api.py调用的封装（同步模式）,简化api调用方式
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: float = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._use_async = False
        self._client = None

    @property
    def client(self):
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=self.timeout,
                proxies={
                    # do not use proxy for locahost
                    "all://127.0.0.1": None,
                    "all://localhost": None,
                },
            )
        return self._client

    def post(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        url = self.base_url + url
        while retry > 0:
            try:
                if stream:
                    return self.client.stream(
                        "POST", url, data=data, json=json, **kwargs
                    )
                else:
                    return self.client.post(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when post {url}: {e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                retry -= 1

    def list_knowledge_bases(self):
        """
        对应api.py/knowledge_base/list_knowledge_bases接口
        """
        response = self.post("/knowledge_base/list_knowledge_bases")
        return response.json().get("data", [])

    def create_knowledge_base(self, knowledge_base_name: str, vector_store_type: str):
        """
        对应api.py/knowledge_base/create_knowledge_base接口
        """
        data = {
            "knowledge_base_name": knowledge_base_name,
            "vector_store_type": vector_store_type,
        }

        response = self.post(
            "/knowledge_base/create_knowledge_base",
            json=data,
        )
        return response.json()

    def clear_knowledge_base(self, knowledge_base_name: str):
        """
        对应api.py/knowledge_base/clear_knowledge_base接口
        """
        response = self.post(
            "/knowledge_base/clear_knowledge_base",
            json=f"{knowledge_base_name}",
        )
        return response.json()

    def upload_kb_docs(
        self,
        files: List[Union[str, Path, bytes]],
        knowledge_base_name: str,
        override: bool = True,
    ):
        """
        对应api.py/knowledge_base/upload_docs接口
        """

        def convert_file(file, filename=None):
            if isinstance(file, bytes):  # raw bytes
                file = BytesIO(file)
            elif hasattr(file, "read"):  # a file io like object
                filename = filename or file.name
            else:  # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        files = [convert_file(file) for file in files]
        data = {"knowledge_base_name": knowledge_base_name, "override": override}
        response = self.post(
            "/knowledge_base/upload_docs",
            data=data,
            files=[
                ("files", (filename, file, "application/octet-stream"))
                for filename, file in files
            ],
        )
        return response.json()

    def knowledge_base_chat(
        self,
        query: str,
        knowledge_base_name: str,
        history: List[Dict] = [],
        stream: bool = True,
        return_docs: bool = False,
    ):
        """
        对应api.py/chat/knowledge_base_chat接口
        """
        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "history": history,
            # "topk": 8,
            "stream": stream,
            "return_docs": return_docs,
        }
        url = self.base_url + "/chat/knowledge_base_chat"
        headers = {"Content-Type": "application/json"}
        response = requests.request(
            "POST", url, headers=headers, data=json.dumps(data), stream=True
        )
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                if chunk.startswith("data: "):
                    data = json.loads(chunk[6:].strip())
                elif chunk.startswith(":"):
                    continue
                else:
                    data = json.loads(chunk)
                yield data.get("result")


    def get_kb_details(self, knowledge_base_name: str):
        """
        对应api.py/knowledge_base/get_kb_details接口
        """
        response = self.post(
            "/knowledge_base/get_kb_details",
            json={"knowledge_base_name": knowledge_base_name},
        )
        return response.json()
