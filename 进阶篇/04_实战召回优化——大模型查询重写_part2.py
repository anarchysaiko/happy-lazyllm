# 子问题查询
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

# 子问题拆分提示词
# 用于将主问题拆解为多个子问题，有助于系统从不同角度理解问题
sub_query_prompt = (
    "你是一个查询重写助手，如果用户查询较为抽象或笼统，将其分解为多个角度的具体问题。\
                  注意，你不需要进行回答，只需要根据问题的字面进行子问题拆分，输出不要超过3条。\
                  下面是一个简单的例子：\
                  输入：RAG是什么？\
                  输出：RAG的定义是什么？\
                       RAG是什么领域内的名词？\
                       RAG有什么特点？"
)

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

# 创建子问题生成器
sub_query_generator = llm.share(ChatPrompter(instruction=sub_query_prompt))
sub_queries_str = sub_query_generator(query)
sub_queries = [q.strip() for q in sub_queries_str.split("\n") if q.strip()]
print(f"拆分的子问题：\n{sub_queries}")

# 对每个子问题进行检索并合并结果
all_context = []
for sub_query in sub_queries:
    print(f"\n正在处理子问题: {sub_query}")
    # 使用检索器检索与子问题相关的文档
    doc_node_list = retriever(query=sub_query)
    # 将检索到的文档内容添加到上下文列表中
    for node in doc_node_list:
        all_context.append(node.get_content())

# 去重并合并上下文
# 使用set去除重复内容，然后合并为一个字符串
unique_context = list(set(all_context))
context_str = "".join(unique_context)
print(f"\n合并后的上下文长度: {len(context_str)}")

# 将查询和合并后的上下文作为大模型的输入
# 构造输入字典，包含查询和上下文
res = llm(
    {
        "query": query,
        "context_str": context_str,
    }
)

# 输出大模型的回答
print("\n回答: ", res)
