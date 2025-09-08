from lazyllm import OnlineEmbeddingModule


online_embed = OnlineEmbeddingModule(
    source="qwen",
    api_key="sk-这里填写你申请的key",
)


print(
    "online embed: ", online_embed("hello world")
)  # 这里输出字符串hello world的高维向量
