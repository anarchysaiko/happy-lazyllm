import lazyllm
from lazyllm import bind

# 初始化文档对象，指定数据目录和嵌入模型
# 使用Qwen的在线嵌入模型进行文档向量化
docs = lazyllm.Document(
    "./data_kb",
    embed=lazyllm.OnlineEmbeddingModule(
        source="qwen",
        api_key="sk-这里填写你申请的key",
    ),
)

# 创建名为"block"的节点组，使用换行符分割文档内容
# 这样可以将文档按行分割成多个节点，便于后续检索
docs.create_node_group(name="block", transform=(lambda d: d.split("\n")))

# 定义提示词模板，指导模型如何根据文档内容回答问题
prompt = "你是一个知识问答助手，请通过给定的文档回答用户问题。"

# 构建处理管道(pipeline)
with lazyllm.pipeline() as ppl:
    # 并行处理模块，用于同时执行多个检索器
    with lazyllm.parallel().sum as ppl.prl:
        # CoarseChunk是LazyLLM默认提供的大小为1024的分块名
        # 使用BM25算法的中文检索器，从"block"节点组中检索top3相关文档
        ppl.prl.retriever1 = lazyllm.Retriever(
            doc=docs, group_name="block", similarity="bm25_chinese", topk=3
        )
        # 使用余弦相似度的检索器，从"block"节点组中检索top3相关文档
        ppl.prl.retriever2 = lazyllm.Retriever(
            doc=docs, group_name="block", similarity="cosine", topk=3
        )

    # 重排序模块，使用在线重排序模型对检索结果进行重新排序
    # 通过ModuleReranker和Qwen的在线重排序模型，选择最相关的top3文档
    ppl.reranker = lazyllm.Reranker(
        name="ModuleReranker",
        model=lazyllm.OnlineEmbeddingModule(
            source="qwen",
            api_key="sk-这里填写你申请的key",
            type="rerank",
        ),
        topk=3,
    ) | bind(query=ppl.input)

    # 格式化模块，将检索到的节点内容和查询组合成字典格式
    # 为大模型准备输入数据，包含上下文和用户查询
    ppl.formatter = (
        lambda nodes, query: dict(
            context_str="".join([node.get_content() for node in nodes]),
            query=query,
        )
    ) | bind(query=ppl.input)

    # 大模型模块，使用Qwen-plus模型进行问答
    # 配置提示词模板和额外的上下文键
    ppl.llm = lazyllm.OnlineChatModule(
        source="qwen",
        model="qwen-plus-latest",
        api_key="sk-这里填写你申请的key",
    ).prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=["context_str"]))


# 启动Web服务模块，将处理管道部署为Web服务
# 指定端口号为23491，并等待服务启动完成
webpage = lazyllm.WebModule(ppl, port=23491).start().wait()
