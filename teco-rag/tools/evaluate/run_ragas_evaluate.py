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

import os, sys
sys.path.append(os.getcwd())

import argparse

parser = argparse.ArgumentParser(prog='Teco-rag-evaluation',
                                     description='')
parser.add_argument("--llm_server_ip", type=str)
parser.add_argument("--llm_server_port", type=str)
parser.add_argument("--llm_server_grpc_port", default="", type=str)
parser.add_argument("--llm_model_name", type=str)
parser.add_argument("--llm_model_engine", type=str)

parser.add_argument("--embedding_model_name_or_path", type=str)
parser.add_argument("--embedding_model_engine", type=str)

parser.add_argument("--eval_file_path", type=str)
parser.add_argument("--openai_api_key", default=None, type=str)

args = parser.parse_args()

########################################################
# Step 1. 实例化LLM和Embedding
########################################################
if args.openai_api_key is not None:
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory
    llm = llm_factory()
    embedding_model = embedding_factory()
else:
    from rag.connector.utils import get_llm, get_embedding_model
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.embeddings.base import LangchainEmbeddingsWrapper

    llm_kwargs = {"grpc_port": args.llm_server_grpc_port}
    llm = get_llm(args.llm_model_name,
                  args.llm_model_engine,
                  args.llm_server_ip,
                  args.llm_server_port,
                  **llm_kwargs)
    llm = LangchainLLMWrapper(llm)

    embedding_model = get_embedding_model(args.embedding_model_name_or_path,
                                          args.embedding_model_engine)
    embedding_model = LangchainEmbeddingsWrapper(embedding_model)


########################################################
# Step 2. 加载评估数据
########################################################
import json
from datasets import Dataset
with open(args.eval_file_path, 'r') as file:
    json_data = json.load(file)

question, answer, ground_truth, contexts = [], [], [], []
for entry in json_data:
    question.append(entry["question"])
    answer.append(entry["answer"])
    contexts.append(entry["context"])
    ground_truth.append(entry["gt_answer"])


data_samples = {
    'question': question,
    'answer': answer,
    'contexts': contexts,
    'ground_truth': ground_truth
}
dataset = Dataset.from_dict(data_samples)

########################################################
# Step 3. 执行评估
########################################################
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_similarity,
    answer_correctness,
    faithfulness,
    context_recall,
    context_precision,
)

answer_relevancy.llm = llm
answer_relevancy.embeddings = embedding_model
faithfulness.llm = llm
context_precision.llm = llm
context_recall.llm = llm

metrics = [
    context_precision,
    faithfulness,
    answer_relevancy,
    context_recall,
]

result = evaluate(dataset, metrics=metrics)
print(result)
# df = result.to_pandas()
# df.head()

