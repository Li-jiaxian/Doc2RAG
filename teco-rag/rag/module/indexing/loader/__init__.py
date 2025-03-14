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


from .doc_loader import CustomizedOcrDocLoader
from .pdf_loader import CustomizedOcrPdfLoader
from .pptx_loader import CustomizedPPTXLoader
from .plumber_pdf_loader import CustomizedPDFPlumberLoader
from .smart_pdf_loader import CustomizedSmartLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader


LOADER_MAPPING = {
    "CustomizedPPTXLoader": [".pptx"],

    # "UnstructuredFileLoader": [".pdf"],
    # "CustomizedOcrPdfLoader": [".pdf"],
    "CustomizedSmartLoader": [".pdf"],
    # UnstructuredFileLoader: [".pdf", ".txt"],
    "CustomizedOcrDocLoader": [
        ".docx",
    ],  # TODO docx包无法读取doc文件, 要么把doc转成docx, 要么换一个新的加载器
    "UnstructuredWordDocumentLoader": [".doc"],
}
