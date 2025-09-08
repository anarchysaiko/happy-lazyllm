from lazyllm import Document, Retriever, OnlineEmbeddingModule

# 定义 embedding 模型
online_embed = OnlineEmbeddingModule(
    source="qwen",
    api_key="sk-这里填写你申请的key",
)


# 文档加载
docs = Document("./data_kb", embed=online_embed)
docs.create_node_group(name="block", transform=(lambda d: d.split("\n")))

# 定义两个不同的检索器对比是否使用阈值过滤的效果
retriever1 = Retriever(docs, group_name="block", similarity="cosine", topk=6)
retriever2 = Retriever(
    docs, group_name="block", similarity="cosine", similarity_cut_off=0.6, topk=6
)

# 执行检索
query = "2008年参加花样游泳的运动员都有谁？"
result1 = retriever1(query=query)
result2 = retriever2(query=query)

print("未设定 similarity_cut_off：")
print("\n\n".join([res.get_content() for res in result1]))
print("设定 similarity_cut_off=0.6 ：")
print("\n\n".join([res.get_content() for res in result2]))
