# 导入所需的库
import lazyllm

# 导入评估工具：LLMContextRecall（基于大语言模型的上下文召回率）、NonLLMContextRecall（非大语言模型的上下文召回率）、ContextRelevance（上下文相关性）
from lazyllm.tools.eval import LLMContextRecall, NonLLMContextRecall, ContextRelevance

# 导入Hugging Face datasets库，用于加载数据集
from datasets import load_dataset

# 加载cmrc2018数据集，指定缓存目录为当前目录下的datasets文件夹
dataset = load_dataset("cmrc2018", cache_dir="./datasets")


def create_evaluation_data(test_dataset, topk_values=[1, 3, 5]):
    """创建用于评估的数据结构

    Args:
        test_dataset: 测试数据集
        topk_values: 检索器返回的topk值（召回的文档数）列表，默认为[1, 3, 5]

    Returns:
        dict: 包含不同topk值评估数据的字典
    """
    # 文档加载，将data_kb目录作为知识库
    documents = lazyllm.Document(dataset_path="./data_kb")

    # 初始化评估数据字典
    evaluation_data = {}

    # 为每个topk值创建评估数据
    for topk in topk_values:
        # 检索组件定义
        # doc: 指定文档对象
        # group_name: 指定文档分组方式为"CoarseChunk"(粗粒度分块)
        # similarity: 使用"bm25_chinese"算法计算相似度，适合中文检索
        # topk: 返回最相关的topk个结果
        retriever = lazyllm.Retriever(
            doc=documents,
            group_name="CoarseChunk",
            similarity="bm25_chinese",
            topk=topk,
        )

        # 初始化数据列表
        data = []
        # 使用测试集的前10个样本进行评估
        for i in range(min(10, len(test_dataset))):
            # 获取当前样本
            sample = test_dataset[i]
            # 提取问题
            question = sample["question"]
            # 提取答案，取第一个答案作为标准答案
            answer = sample["answers"]["text"][0]
            # 提取标注的应当被召回的段落
            context_reference = [sample["context"]]

            # 获取检索器召回的文档
            doc_node_list = retriever(query=question)
            # 提取召回文档的内容
            context_retrieved = [node.get_content() for node in doc_node_list]

            # 将数据添加到列表中
            data.append(
                {
                    "question": question,
                    "answer": answer,
                    "context_retrieved": context_retrieved,
                    "context_reference": context_reference,
                }
            )

        # 将当前topk值的数据添加到评估数据字典中
        evaluation_data[topk] = data

    return evaluation_data


def evaluate_rag_performance(evaluation_data):
    """评估RAG性能

    Args:
        evaluation_data: 包含不同topk值评估数据的字典

    Returns:
        dict: 包含不同topk值评估结果的字典
    """
    # 初始化评估工具
    # NonLLMContextRecall: 计算召回文档的命中率
    m_recall = NonLLMContextRecall()
    # ContextRelevance: 计算召回文档中的上下文相关性分数
    m_cr = ContextRelevance()
    # LLMContextRecall: 计算基于LLM的召回率
    m_lcr = LLMContextRecall(
        lazyllm.OnlineChatModule(
            source="qwen",
            model="qwen-plus-latest",
            api_key="sk-这里填写你申请的key",
        )
    )

    # 初始化结果字典
    results = {}

    # 对每个topk值进行评估
    for topk, data in evaluation_data.items():
        print(f"\n=== TopK = {topk} 评估结果 ===")

        # 计算召回文档的命中率
        recall_score = m_recall(data)
        print(f"召回率 (Recall): {recall_score}")

        # 计算召回文档中的上下文相关性分数
        relevance_score = m_cr(data)
        print(f"上下文相关性 (Context Relevance): {relevance_score}")

        # 计算基于LLM的召回率
        try:
            llm_recall_score = m_lcr(data)
            print(f"基于LLM的召回率 (LLM Context Recall): {llm_recall_score}")
        except Exception as e:
            print(f"计算基于LLM的召回率时出错: {e}")
            llm_recall_score = 0.0

        # 将结果添加到结果字典中
        results[topk] = {
            "recall": recall_score,
            "relevance": relevance_score,
            "llm_recall": llm_recall_score,
        }

    return results


def main():
    # 创建评估数据
    print("正在创建评估数据...")
    evaluation_data = create_evaluation_data(dataset["test"], topk_values=[1, 3, 5])

    # 评估RAG性能
    print("开始评估RAG性能...")
    results = evaluate_rag_performance(evaluation_data)

    # 打印最终结果
    print("\n=== 最终评估结果 ===")
    for topk, scores in results.items():
        print(f"\nTopK = {topk}:")
        print(f"  召回率: {scores['recall']:.4f}")
        print(f"  上下文相关性: {scores['relevance']:.4f}")
        print(f"  基于LLM的召回率: {scores['llm_recall']:.4f}")


if __name__ == "__main__":
    main()
