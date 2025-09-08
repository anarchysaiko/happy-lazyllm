from lazyllm import OnlineChatModule, LLMParser, Document, Retriever

# 创建在线大模型对象，指定来源、模型和API密钥
llm = OnlineChatModule(
    source="qwen",
    model="qwen-plus-latest",
    api_key="sk-这里填写你申请的key",
    stream=False,
)  # 调用大模型

# LLMParser 是 LazyLLM 内置的基于 LLM 进行节点组构造的类，支持 summary，keywords和qa三种
summary_llm = LLMParser(llm, language="zh", task_type="summary")  # 摘要提取LLM
keyword_llm = LLMParser(llm, language="zh", task_type="keywords")  # 关键词提取LLM
qapair_llm = LLMParser(llm, language="zh", task_type="qa")  # 问答对提取LLM


# 利用 LLMParser 创建节点组
docs = Document("./data_kb")

docs.create_node_group(
    name="summary", transform=lambda d: summary_llm(d), trans_node=True
)
docs.create_node_group(
    name="keyword", transform=lambda d: keyword_llm(d), trans_node=True
)
docs.create_node_group(
    name="qapair", transform=lambda d: qapair_llm(d), trans_node=True
)

# 查看节点组内容，此处我们通过一个检索器召回一个节点并打印其中的内容，后续都通过这个方式实现
group_names = ["CoarseChunk", "summary", "keyword", "qapair"]
for group_name in group_names:
    retriever = Retriever(
        docs, group_name=group_name, similarity="bm25_chinese", topk=1
    )
    node = retriever("亚硫酸盐有什么作用？")
    print(f"======= {group_name} =====")
    print(node[0].get_content())
