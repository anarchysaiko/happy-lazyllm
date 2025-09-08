from lazyllm import OnlineEmbeddingModule, Document, Retriever


online_embed = OnlineEmbeddingModule(
    source="qwen",
    api_key="sk-这里填写你申请的key",
)


# 文档加载
docs = Document("./data_kb", embed=online_embed)
docs.create_node_group(name="block", transform=(lambda d: d.split("\n")))

# 定义两个不同的检索器在同一个节点组使用不同相似度方法进行检索
retriever1 = Retriever(docs, group_name="block", similarity="cosine", topk=3)
retriever2 = Retriever(docs, group_name="block", similarity="bm25_chinese", topk=3)

# 执行查询
query = "都有谁参加了2008年的奥运会？"
result1 = retriever1(query=query)
result2 = retriever2(query=query)

print("余弦相似度召回结果：")
print("\n\n".join([res.get_content() for res in result1]))
print("bm25召回结果：")
print("\n\n".join([res.get_content() for res in result2]))
