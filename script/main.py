import json
import os
import sys
import time
import traceback
from typing import List

import httpx
from pydantic import BaseModel, Field
from tqdm import tqdm

from utils import logger

OUT_DIR = "../teco-rag/eval"


class InputSchema(BaseModel):
    id: int = Field(..., description="The id of the data")
    question: str = Field(..., description="The data to be processed")


class OutputSchema(BaseModel):
    id: int = Field(..., description="The id of the data")
    question: str = Field(..., description="The data to be processed")
    answer: str = Field(..., description="The answer to the question")


def request_to_teco(input_li: List[InputSchema]) -> List[OutputSchema]:
    url = "http://127.0.0.1:7861/chat/knowledge_base_chat"
    headers = {"Content-Type": "application/json"}

    output_li = []
    for input_item in tqdm(input_li, desc="处理问题", file=sys.stdout):
        start_time = time.time()
        score_threshold = 0
        try:
            payload = {
                "query": input_item.question,
                "knowledge_base_name": "test_6",
                "history": [],
                "stream": False,
                "return_docs": True,
            }

            logger.info(f"发送请求: ID={input_item.id}, 问题='{input_item.question}'")

            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=payload, timeout=300)

            if response.status_code == 200:
                result = json.loads(response.text[6:])
                answer = result.get("result", "请求成功，但未返回答案").strip()
                docs = result.get("docs", [])
                # 对文档按相似度得分进行排序
                sorted_docs = sorted(
                    docs, key=lambda x: x.get("similarity_score", 0), reverse=True
                )

                for doc in sorted_docs:
                    log_message = f"ID={input_item.id} 文档: {doc}, 得分: {doc.get('similarity_score')}"
                    if doc.get("similarity_score") >= score_threshold:
                        logger.info(log_message)
                    else:
                        logger.warning(log_message)
                logger.info(f"ID={input_item.id} 的答案: {answer}")
            else:
                answer = f"请求失败，状态码：{response.status_code}"
                logger.error(
                    f"请求失败: ID={input_item.id}, 状态码={response.status_code}"
                )

        except Exception as e:
            error_message = (
                f"处理请求时发生错误: ID={input_item.id}\n{traceback.format_exc()}"
            )
            logger.error(error_message)
            answer = f"处理请求时发生错误: {str(e)}"

        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info(f"请求完成: ID={input_item.id}, 耗时={elapsed_time:.2f}秒")

        output_li.append(
            OutputSchema(id=input_item.id, question=input_item.question, answer=answer)
        )

    logger.info(f"处理完成，共处理 {len(output_li)} 个问题")

    return output_li


def main():
    input_file_path = "input/eval_data.json"
    q_id_list = []
    # q_id_list = [11, 76, 150] # 11总结的内容会吧3.8这种去掉, 很怪
    q_id_list: List[int] = [2, 3, 16, 8, 11, 12, 61, 120, 121, 127] # 政策相关
    # q_id_list = range(158, 241) # 政策相关问题
    with open(input_file_path, "r") as f:
        question_list = json.load(f)
        input_li = [
            InputSchema(id=q["id"], question=q["question"])
            for q in question_list
            if not q_id_list or q["id"] in q_id_list
        ]
        input_li.sort(key=lambda x: x.id)  # 根据id从小到大排序

    start_time = time.time()
    output_li = request_to_teco(input_li)
    end_time = time.time()
    total_time = end_time - start_time

    logger.info(f"所有请求处理完成，总耗时：{total_time:.2f}秒")

    # 创建输出目录
    # os.makedirs(OUT_DIR, exist_ok=True)
    # logger.info(f"创建输出目录: {OUT_DIR}")
    # with open(os.path.join(OUT_DIR, "predict.json"), "w") as f:
    #     logger.info(f"写入文件: {os.path.join(OUT_DIR, 'predict.json')}")
    #     json.dump(
    #         [output.model_dump() for output in output_li],
    #         f,
    #         indent=2,
    #         ensure_ascii=False,
    #     )


if __name__ == "__main__":
    main()
    # 2023滨湖区机器人产业研讨峰会有哪些政府领导参加
