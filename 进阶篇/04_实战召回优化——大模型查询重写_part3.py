# 分步骤查询
# 导入必要的库
import lazyllm
from lazyllm import Document, ChatPrompter, Retriever

# prompt设计
# 查询改写提示词，用于将用户的查询改写得更加清晰，不回答问题
rewrite_prompt = "你是一个查询改写助手，将用户的查询改写的更加清晰。\
                注意，你不需要对问题进行回答，只需要对原始问题进行改写。\
                下面是一个简单的例子：\
                输入：RAG\
                输出：为我介绍下RAG。\
                    \
                用户输入为："

# 判别提示词，用于判断某个回答能否解决对应的问题
judge_prompt = (
    "你是一个判别助手，用于判断某个回答能否解决对应的问题。如果回答可以解决问题，则输出True，否则输出False。\
                注意，你的输出只能是True或者False。不要带有任何其他输出。\
                当前回答为{context_str} \n"
)

# AI问答助手提示词，用于根据上下文和问题提供答案
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

# 创建查询改写器
rewrite_robot = llm.share(ChatPrompter(instruction=rewrite_prompt))

# 创建AI问答助手
robot = llm.share(ChatPrompter(instruction=robot_prompt, extra_keys=["context_str"]))

# 创建判别助手
judge_robot = llm.share(
    ChatPrompter(instruction=judge_prompt, extra_keys=["context_str"])
)

# 定义查询问题
query = "MIT OpenCourseWare是啥？"

# 分步骤查询逻辑
LLM_JUDGE = False
while LLM_JUDGE is not True:
    # 执行查询重写
    query_rewrite = rewrite_robot(query)
    print(f"\n重写的查询：{query_rewrite}")

    # 获取重写后的查询结果
    doc_node_list = retriever(query_rewrite)
    # 使用AI问答助手生成回答
    res = robot(
        {
            "query": query_rewrite,
            "context_str": "\n".join([node.get_content() for node in doc_node_list]),
        }
    )

    # 判断当前回复是否能满足query要求
    LLM_JUDGE = bool(judge_robot({"query": query, "context_str": res}))
    print(f"\nLLM判断结果：{LLM_JUDGE}")

# 打印最终结果
print("\n最终回复: ", res)
