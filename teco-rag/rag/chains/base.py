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


"""Base interface that all RAG applications should implement."""

from abc import ABC, abstractmethod
from typing import Generator


class BaseIndexingChain(ABC):

    @abstractmethod
    def load(self, file, loader_name):
        pass

    @abstractmethod
    def split(self, docs, splitter):
        pass

    @abstractmethod
    def store(self, file, chunks):
        pass


class BaseRetrievalChain(ABC):
    """
    检索链的抽象基类。

    定义了检索过程中的主要步骤,包括预检索处理、检索和后检索处理。
    所有具体的检索链实现都应该继承这个基类并实现其抽象方法。
    """

    @abstractmethod
    def pre_retrieval(self, query):
        pass

    @abstractmethod
    def retrieval(self, query):
        pass

    @abstractmethod
    def post_retrieval(self, query, docs):
        pass


class BaseGenerationChain(ABC):

    @abstractmethod
    def augment(self, query, docs):
        pass

    @abstractmethod
    def generate(self, prompt):
        pass

