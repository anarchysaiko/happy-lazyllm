# 导入所需的库
import os
import lazyllm
from datasets import load_dataset
from lazyllm import bind


# 加载cmrc2018数据集，指定缓存目录为当前目录下的datasets文件夹
# cmrc2018是一个中文机器阅读理解数据集
dataset = load_dataset("cmrc2018", cache_dir="./datasets")  # 指定下载路径
print(dataset)  # 打印数据集信息


# 构建知识库的函数
def create_KB(dataset):
    """基于测试集中的context字段创建一个知识库，每10条数据为一个txt，最后不足10条的也为一个txt"""
    # 提取数据集中的context字段内容
    Context = []
    for i in dataset:
        Context.append(i["context"])
    Context = list(set(Context))  # 去重后获得256个语料

    # 计算需要的文件数，每10条数据为一组
    chunk_size = 10
    total_files = (len(Context) + chunk_size - 1) // chunk_size  # 向上取整

    # 创建文件夹data_kb保存知识库语料
    os.makedirs("data_kb", exist_ok=True)

    # 按 10 条数据一组写入多个文件
    for i in range(total_files):
        chunk = Context[i * chunk_size : (i + 1) * chunk_size]  # 获取当前 10 条数据
        file_name = f"./data_kb/part_{i + 1}.txt"  # 生成文件名
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("\n".join(chunk))  # 以换行符分隔写入文件

        print(f"文件 {file_name} 写入完成！")  # 提示当前文件已写入


# 调用create_KB()函数，使用测试集数据创建知识库
create_KB(dataset["test"])  # 调用create_KB()创建知识库

# 展示其中一个txt文件中的内容，验证知识库创建是否成功
with open("data_kb/part_1.txt") as f:
    print(f.read())


# 文档加载，将创建的知识库文件夹作为数据源
documents = lazyllm.Document(dataset_path="./data_kb")

# 定义提示词模板，告诉模型扮演AI问答助手的角色
prompt = "You will act as an AI question-answering assistant and complete a dialogue task. \
          In this task, you need to provide your answers based on the given context and questions."

# 创建一个处理流程(pipeline)，包含检索和生成两个主要步骤
with lazyllm.pipeline() as ppl:
    # 检索组件定义：用于从知识库中检索相关信息
    # doc: 指定文档对象
    # group_name: 指定文档分组方式为"CoarseChunk"(粗粒度分块)
    # similarity: 使用"bm25_chinese"算法计算相似度，适合中文检索
    # topk: 返回最相关的3个结果
    ppl.retriever = lazyllm.Retriever(
        doc=documents, group_name="CoarseChunk", similarity="bm25_chinese", topk=3
    )

    # 格式化组件：将检索到的节点内容和查询问题格式化为适合LLM处理的格式
    # nodes: 检索到的文档节点
    # query: 用户的查询问题
    # context_str: 将所有检索到的节点内容拼接成字符串
    ppl.formatter = (
        lambda nodes, query: {
            "query": query,
            "context_str": "".join([node.get_content() for node in nodes]),
        }
    ) | bind(query=ppl.input)

    # 生成组件定义：使用在线大语言模型进行回答生成
    # source: 指定模型来源为"qwen"(通义千问)
    # model: 指定具体模型为"qwen-plus-latest"
    # api_key: 设置API密钥用于访问模型
    # prompt: 使用ChatPrompter包装提示词模板，并添加额外的context_str上下文信息
    ppl.llm = lazyllm.OnlineChatModule(
        source="qwen",
        model="qwen-plus-latest",
        api_key="sk-这里填写你申请的key",
    ).prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=["context_str"]))

# 启动Web服务模块，提供图形化界面进行交互
# ppl: 传入上面定义的处理流程
# port: 指定服务运行端口为23466
lazyllm.WebModule(ppl, port=23466).start().wait()
