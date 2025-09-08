from lazyllm.tools.rag import Document, DocNode, Retriever
from bs4 import BeautifulSoup
import re
from pathlib import Path
import lazyllm
from lazyllm import bind


def clean_html_text(html_input, extra_info=None):
    """
    清理HTML文本，去除标签和多余的空白字符，并将其封装为DocNode对象
    :param html_input: 包含HTML标签的原始文本或文件路径
    :param extra_info: 额外的元数据信息
    :return: 包含DocNode对象的列表
    """
    # 如果输入是路径，则读取文件内容
    if isinstance(html_input, Path):
        with open(html_input, "r", encoding="utf-8") as f:
            html_content = f.read()
    else:
        html_content = html_input

    # 使用 BeautifulSoup 去除 HTML 标签，只保留纯文本内容
    soup = BeautifulSoup(html_content, "html.parser")
    clean_text = soup.get_text()

    # 去除多余的换行符和空白字符
    clean_text = re.sub(r"\n+", "\n", clean_text)  # 将多个连续的换行符替换为单个换行符
    clean_text = re.sub(r"\s+", " ", clean_text)  # 将多个连续的空白字符替换为单个空格

    # 创建 DocNode 对象并返回列表
    node = DocNode(text=clean_text, metadata=extra_info or {})
    return [node]


# 创建 Document 实例，指定数据集路径
rag_data_path = "./rag_data"
documents = Document(dataset_path=rag_data_path)
# 为 Document 实例添加自定义 HTML reader
documents.add_reader("*.html", clean_html_text)

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
    ppl.retriever = Retriever(
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
