# 查询扩写
# 导入必要的库
import lazyllm
from lazyllm import Document, ChatPrompter, Retriever

# 查询改写提示词
# 用于将用户的查询改写得更加清晰，不回答问题
rewrite_prompt = "你是一个查询改写助手，将用户的查询改写的更加清晰。\
                注意，你不需要对问题进行回答，只需要对原始问题进行改写。\
                下面是一个简单的例子：\
                输入：RAG\
                输出：为我介绍下RAG。\
                    \
                用户输入为："

# AI问答助手提示词
# 用于根据上下文和问题提供答案
robot_prompt = "你是一个友好的 AI 问答助手，你需要根据给定的上下文和问题提供答案。\
                根据以下资料回答问题：\
                {context_str} \n"

# 加载文档库，定义检索器和在线大模型
# 创建文档对象，指定数据集路径
documents = Document(dataset_path="./data_kb")  # 请在 dataset_path 传入数据集绝对路径
# 创建检索器，指定文档、分组名、相似度算法和返回结果数量
retriever = Retriever(
    doc=documents, group_name="CoarseChunk", similarity="bm25_chinese", topk=3
)  # 定义检索组件
# 创建在线大模型对象，指定来源、模型和API密钥
llm = lazyllm.OnlineChatModule(
    source="qwen",
    model="qwen-plus-latest",
    api_key="sk-这里填写你申请的key",
)  # 调用大模型

# 设置大模型的提示词
llm.prompt(ChatPrompter(instruction=robot_prompt, extra_keys=["context_str"]))
# 定义查询问题
query = "MIT OpenCourseWare是啥？"

# 创建查询改写器并改写查询
query_rewriter = llm.share(ChatPrompter(instruction=rewrite_prompt))
query = query_rewriter(query)
print(f"改写后的查询：\n{query}")

# 使用检索器检索相关文档
doc_node_list = retriever(query=query)

# 将查询和召回节点中的内容组成dict，作为大模型的输入
# 构造大模型输入参数，包括查询和上下文
res = llm(
    {
        "query": query,
        "context_str": "".join([node.get_content() for node in doc_node_list]),
    }
)

# 输出大模型的回答
print("\n回答: ", res)
