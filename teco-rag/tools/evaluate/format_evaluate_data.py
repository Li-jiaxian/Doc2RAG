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
import argparse
import requests

parser = argparse.ArgumentParser(prog='Teco-rag',
                                     description='')
parser.add_argument("--ip", type=str)
parser.add_argument("--port", type=str)
parser.add_argument("--knowledge_base_name", type=str)
parser.add_argument("--org_file_path", type=str)
parser.add_argument("--dump_file_path", type=str)

args = parser.parse_args()


########################################################
# Step 1. 加载原始数据（json格式，只包含filename、question、gt_context、gt_answer信息）
########################################################
f = open(args.org_file_path)
data = json.load(f)
print(data[0])

########################################################
# Step 2. 调用知识对话接口，得到context、answer
########################################################
url = "http://" + args.ip + ":" + args.port + "/chat/knowledge_base_chat"  # 接口URL
headers = {
  'Content-Type': 'application/json'
}
payload = {"query": "", "knowledge_base_name": args.knowledge_base_name, "history": [], "stream": False, "return_docs": True}  # 请求参数
for d in data:
    payload.update({"query": d["question"]})
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    result = json.loads(response.text.split("data: ")[-1])
    # context = "\n".join([doc["context"] for doc in result["docs"]])
    context = [doc["context"] for doc in result["docs"]]
    answer = result["result"]
    d.update({"context": context, "answer": answer})

########################################################
# Step 3. 将数据写回文件
########################################################
with open(args.dump_file_path, 'w') as f:
    json.dump(data, f, ensure_ascii=False)
