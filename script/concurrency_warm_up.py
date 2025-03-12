# concurrency_warm_up.py
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Generator, List

from teco_client_toolkits import ApiType, ClientRequest, TritonRequestParams


def build_input(question):
    _prompt = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{Input}<|im_end|>\n<|im_start|>assistant\n"""
    prompt = _prompt.format(History_input="", History_output="", Input=question)
    return prompt


def run_in_thread_pool(func: Callable, params: List[Dict] = []) -> Generator:
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params:
            tasks.append(pool.submit(func, **kwargs))
        for obj in as_completed(tasks):
            yield obj.result()


def get_parser():
    parser = argparse.ArgumentParser(prog="Teco-LLM-Infer_Concurrency_WarmUp")
    parser.add_argument("--ip", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--qps", type=int)
    return parser.parse_args()


def call_teco_llm_infer(question, q_id, ip, port):
    client = ClientRequest(ip=ip, port=port)
    param = TritonRequestParams(
        mode="non-ensemble",
        max_new_tokens=1024,
        start_id=1,
        end_id=151643,
        topk=1,
        topp=0,
        temperature=1.0,
        len_penalty=1.0,
        repetition_penalty=1.0,
        stop_words_list=[["<|im_end|>"]],
        protocol="grpc",
    )
    result = client.request(
        prompts=build_input(question),
        api_type=ApiType.TRITON,
        stream=True,
        params=param,
    ).streamer
    res = ""
    for out in result:
        res = out["outputs"]
    return res, q_id


questions = """
user: <指令>你是太初（无锡）电子科技有限公司的资深员工，根据已知信息，简洁和专业的来回答问题。请你先判断有多少个问题, 若问题包含多个子问题，每个子问题都要给出答案。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。</指令>\n已知信息：能力达到L1级别
特授予其
白银认证证书及相关标认证使用权（一年）
有效期：2024年01月-2025年01月
龙芯中科
Tecorigin
产品兼容性认证
龙架构兼容互认证书
ProductCompatibilityCertificate
<CERTIFICATE>
LoongArch龙架构
元基系列智能加速卡
经北京百川智能科技有限公司与太初（无铺）电子科技有限公司的联
合严格测试，得出以下结论：
产品兼容性证明
在龙芯3C5000平台上完成美容性酒试，功能与格定性良好。
北京百川智能料技术有限公司的基列大模型与太初（无
特授予龙除构道容互认证书。
）电子科技有限公司的远基系列财加速已完成全适配，产品
功能符合兼容性要求，整体活行稳定，性能表现良好，可以满足实际
经百度飞桨与太初（无锡）电子科技的联合严格测试，得出以下结论：
应用需求，
太初（无提）电子科技有限公司的元系列智能加速卡与百度飞案完成川级兼容
性测试（基于训练），能够达到兼容性要求，整体运行稳定，可以满足用户的
应用需求。
百川认证
产品兼容性认证
ProductCompatibilityCertificate
<CERTIFICATE>
经北京百川智能科技有限公司与太初（无锡）电子科技有限公司的联
合严格测试，得出以下结论：
北京百川智能科技术有限公司的Baichuan系列
大模型与太初（无
锡）电子科技有限公司的元基系列智能加速卡已完成全面适配，产品
功能符合兼容性要求，整体运行稳定，性能表现良好，可以满足实际
应用需求。
日期：2023年11月15日
百川智能
Tecorigin
BAICHUANAI
太初元碁
北京百川智能科技术有限公司
太初（无锡）电子科技有限公司
移动云
Tecorigin
太初元基
产品兼容互认证明
中移（苏州）软件技术有限公司与太初（无锡）电子科技有限
公司共同严格测试，得出结论如下：
E-ECV6.0.0与太初（无锡）电子科技有限公司元基系列AI加速卡
完成适配测试，兼容性良好，可提供安全可靠的AI加速能力。
特此证明！
麻件技术
苏州
太初（无锡）电子科技有限公司
中移（苏州）软件技术有限公司
32
2024年4月
2024年4月
无锡
大初 \n问题： 太初元碁智能加速卡与百度飞桨完成的兼容性测试等级是多少 \n答案：
"""

if __name__ == "__main__":
    args = get_parser()
    params = [
        {"question": questions, "ip": args.ip, "port": args.port, "q_id": i + 1}
        for i in range(args.qps)
    ]
    for res, q_id in run_in_thread_pool(call_teco_llm_infer, params=params):
        print(q_id, res)
