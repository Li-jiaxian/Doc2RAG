import os
import traceback

import httpx

from utils import logger


class FileUploader:
    def __init__(self, base_url: str = "http://127.0.0.1:7861"):
        self.base_url = base_url
        # self.headers = {"Content-Type": "application/json"}
        self.kb_name = "test_6"

    def _upload_file(self, file_path: str) -> str:
        url = f"{self.base_url}/knowledge_base/upload_docs"

        try:
            with open(file_path, "rb") as file:
                files = {
                    "files": (
                        os.path.basename(file_path),
                        file,
                        "application/octet-stream",
                    )
                }
                with httpx.Client() as client:
                    response = client.post(
                        url,
                        files=files,
                        timeout=600,
                        data={
                            "knowledge_base_name": self.kb_name,
                            # "chunk_size": 400,
                            # "chunk_overlap": 100,
                        },
                    )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"成功上传文件: {file_path}")
                    return result.get("message", "文件上传成功")
                else:
                    logger.error(
                        f"上传文件失败: {file_path}, 状态码: {response.status_code}"
                    )
                    return f"上传失败，状态码：{response.status_code}"
        except Exception as e:
            error_message = f"上传文件时发生错误: {file_path}\n{traceback.format_exc()}"
            logger.error(error_message)
            return f"上传文件时发生错误: {str(e)}"

    def upload_files(self, file_dir: str):
        from tqdm import tqdm

        def get_file_count(dir_path: str) -> int:
            count = 0
            for root, _, files in os.walk(dir_path):
                count += len(files)
            return count

        def upload_recursive(current_dir: str, pbar: tqdm):
            for item in os.listdir(current_dir):
                item_path = os.path.join(current_dir, item)
                if os.path.isfile(item_path):
                    result = self._upload_file(item_path)
                    logger.info(result)
                    pbar.update(1)
                elif os.path.isdir(item_path):
                    upload_recursive(item_path, pbar)

        total_files = get_file_count(file_dir)
        with tqdm(total=total_files, desc=file_dir.split("/")[-1]) as pbar:
            upload_recursive(file_dir, pbar)


if __name__ == "__main__":
    # 20210618 甬经信数经〔2021〕93号 - 宁波市加快集成电路产业发展的若干政策.doc
    # FileUploader().upload_files(file_dir="data")
    # FileUploader()._upload_file(file_path="data/组件用户手册/性能优化手册-算子篇_v1.1.0.pdf")
    # FileUploader()._upload_file(file_path="data/政策/20220531 合政办〔2022〕18号 - 合肥市加快推进集成电路产业发展若干政策.pdf")
    # FileUploader()._upload_file(
    #     file_path="data/公司彩页/太初品牌介绍（新）.pdf"
    # )
    # file_txt/20220531 合政办〔2022〕18号 - 合肥市加快推进集成电路产业发展若干政策.pdf.txt
    # FileUploader()._upload_file(file_path="data/视频教学配套PPT/TecoPaddle/02 TecoPaddle-安装-Conda.pptx")
    # FileUploader()._upload_file(file_path="data/内刊/第五期_7-8.pdf")

    # FileUploader().upload_files(file_dir="data/公司彩页")  # 复杂pdf
    # FileUploader().upload_files(file_dir="data/内刊/")  # 双栏pdf
    # FileUploader().upload_files(file_dir="data/视频教学配套PPT/")  # PPT, 尽量一页就是一个chunk, 因为一页PPT通常就只做一件事
    # FileUploader().upload_files(file_dir="data/证书")  # 基本上纯OCR就行
    # FileUploader().upload_files(file_dir="data/政策")  # pdf和doc, docx都有, 有一定规律, 可以定制化解析
    FileUploader().upload_files(file_dir="data/组件用户手册")  # pdf, 有很强规律可以定制化解析
    # FileUploader()._upload_file(file_path="data/证书/产品兼容互认证明-太初&移动云.pdf")
    # FileUploader()._upload_file(file_path="data/组件用户手册/SDAA C编程指南_v1.11.0.pdf")
    # FileUploader()._upload_file(file_path="data/公司彩页/【Teco彩页】1(5).pdf")
