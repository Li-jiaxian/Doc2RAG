群公告
赛事介绍录屏：
https://www.bilibili.com/video/BV1S6mKYdEAd
Teco-RAG参考文档：
https://gitee.com/tecorigin/teco-generative-ai/blob/master/teco-rag/README.md
参赛指南：
https://gitee.com/tecorigin/teco-generative-ai/blob/master/teco-rag/docs/openatom_competition_guide.md
------------------
大模型参赛id：
https://docs.qq.com/sheet/DZUZlZkttY1pESkxO?is_no_hook_redirect=1
数据集&评测问题集网盘链接：https://pan.baidu.com/s/1Z6HftAe2ixqwZN-xqBx0UA?pwd=pwr0 
提取码：pwr0tiktoken


# 显卡信息

teco-smi

# embedding模型下载
```
git clone https://hf-mirror.com/chuxin-llm/Chuxin-Embedding /share_data/models/Chuxin-Embedding
```

# rerank 模型下载
```
git clone https://hf-mirror.com/BAAI/bge-reranker-large /share_data/models/bge-reranker-large
```

# TODO
- [ ] 原来的rag流程是什么, 比如向量 rag 多少个, 最终rerank 
- [ ] 人工标注真实值, 查看具体分数低的问题在哪
- [ ] 关键词基于BM25的多路召回: 从向量数据库中查询document信息, 就可以用于构建我的BM25的关键词召回, 组合到一起再进行RERANK
    - [ ] https://www.perplexity.ai/search/milvus-de-hybrid-searchzui-jia-9g6QLzOMQLmoQAfNKwT85g
    - https://milvus.io/blog/introducing-pymilvus-integrations-with-embedding-models.md
    - https://python.langchain.com/docs/integrations/components/
    - 临时方案, 从数据库查询全部文字, 进行 BM模型训练, 然后再更新向量数据库中的稀疏向量