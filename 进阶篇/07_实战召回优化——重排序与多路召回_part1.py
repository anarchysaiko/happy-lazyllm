from lazyllm import Document, Retriever, OnlineEmbeddingModule, Reranker

# 定义 embedding 模型
embedding_model = OnlineEmbeddingModule(
    source="qwen",
    api_key="sk-这里填写你申请的key",
)

rerank_model = OnlineEmbeddingModule(
    source="qwen",
    api_key="sk-这里填写你申请的key",
    type="rerank",
)


docs = Document("./data_kb", embed=embedding_model)
docs.create_node_group(name="block", transform=(lambda d: d.split("\n")))

# 定义检索器
retriever = Retriever(docs, group_name="block", similarity="cosine", topk=3)

# 定义重排器
# 指定 reranker1 的输出为字典，包含 content, embedding 和 metadata 关键字
# reranker = Reranker('ModuleReranker', model=online_rerank, topk=3, output_format='dict')
# 指定 reranker2 的输出为字符串，并进行串联，未进行串联时输出为字符串列表
reranker = Reranker(
    "ModuleReranker", model=rerank_model, topk=3, output_format="content", join=True
)

# 执行推理
query = "猴面包树有哪些功效？"
result1 = retriever(query=query)
result2 = reranker(result1, query=query)

print("余弦相似度召回结果：")
print("\n\n".join([res.get_content() for res in result1]))
print("开源重排序模型结果：")
print(result2)
