import lazyllm
from lazyllm import Document, Retriever, OnlineChatModule

# 创建在线大模型对象，指定来源、模型和API密钥
llm = OnlineChatModule(
    source="qwen",
    model="qwen-plus-latest",
    api_key="sk-这里填写你申请的key",
)  # 调用大模型

prompt = "你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
          根据以下资料回答问题：\
          {context_str} \n "

robot = llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=["context_str"]))

# 文档加载
docs = Document("./data_kb")
docs.create_node_group(
    name="sentences",
    transform=(lambda d: d.split("\n") if d else ""),
    parent=Document.CoarseChunk,
)

# 定义两个不同的检索器在同一个节点组使用不同相似度方法进行检索
retriever = Retriever(docs, group_name="sentences", similarity="bm25_chinese", topk=3)

# 执行查询
query = "都有谁参加了2008年的奥运会？"

# 原节点检索结果
doc_node_list = retriever(query=query)
doc_node_res = "".join([node.get_content() for node in doc_node_list])
print(f"原节点检索结果：\n{doc_node_res}")
print("=" * 100)

# 父节点对应结果
parent_list = [node.parent.get_text() for node in doc_node_list]
print(f"父节点检索结果：\n{''.join(parent_list)}")
print("=" * 100)

# 将query和召回节点中的内容组成dict，作为大模型的输入
res = robot(
    {"query": query, "context_str": "".join([node_text for node_text in parent_list])}
)

print("系统答案：\n", res)
