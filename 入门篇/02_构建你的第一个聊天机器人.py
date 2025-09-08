import lazyllm

# 使用在线模型
llm = lazyllm.OnlineChatModule(
    source="qwen",
    model="qwen-plus-latest",  # 使用一个标准的模型名称
    api_key="sk-这里填写你申请的key",
)

lazyllm.WebModule(llm, port=23466).start().wait()
