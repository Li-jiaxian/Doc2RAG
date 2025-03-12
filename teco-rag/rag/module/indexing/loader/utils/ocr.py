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


import re
from typing import Any, Optional

# 全局变量用于缓存 RapidOCR 实例
_rapid_ocr_instance: Optional[Any] = None


def get_rapid_ocr(use_cuda: bool = True) -> Any:
    """获取 RapidOCR 实例，支持 CUDA 加速

    Args:
        use_cuda (bool): 是否使用 CUDA 加速。默认为 True。

    Returns:
        Any: RapidOCR 实例对象
    """
    global _rapid_ocr_instance

    if _rapid_ocr_instance is not None:
        return _rapid_ocr_instance

    try:
        from rapidocr_paddle import RapidOCR

        _rapid_ocr_instance = RapidOCR(
            det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda
        )
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR

        _rapid_ocr_instance = RapidOCR()

    return _rapid_ocr_instance


def filter_text(text: str, max_length: int = 30) -> bool:
    """过滤文本内容，对于长文本要求必须包含中文字符

    Args:
        text (str): 待过滤的文本内容

    Returns:
        bool: 如果文本符合要求返回 True，否则返回 False
    """
    if len(text) > max_length:
        return bool(re.search(r'[\u4e00-\u9fff]+', text)) # 判断是否包含中文
    return True

def extract_from_images_with_rapidocr(
    images,
) -> str:
    """使用 RapidOCR 从图片中提取文本

    Args:
        images: 需要提取文本的图片列表

    Returns:
        str: 从图片中提取的文本内容，多个文本片段以换行符分隔

    Raises:
        ImportError: 如果未安装 rapidocr-onnxruntime 包
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        raise ImportError(
            "`rapidocr-onnxruntime` package not found, please install it with "
            "`pip install rapidocr-onnxruntime`"
        )
    ocr = get_rapid_ocr()
    all_text = ""
    for img in images:
        try:
            result, _ = ocr(img)
            if result:
                for res in result:
                    _, text, score = res

                    if score > 0.6 and filter_text(text):
                        all_text += "\n" + text
        except Exception as e:
            print(e)
    return all_text
