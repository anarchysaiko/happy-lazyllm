# 导入 lazyllm 库，这是一个大语言模型开发工具包
import lazyllm

# 从 lazyllm 中导入 pipeline 模块，用于构建处理流程
from lazyllm import pipeline

# 从 lazyllm.tools 中导入 IntentClassifier 模块，用于意图识别和分类
from lazyllm.tools import IntentClassifier

# 创建基础的在线聊天模型实例（大语言模型）
# source="qwen" 表示使用通义千问系列模型
# model="qwen-plus-latest" 表示使用 qwen-plus 的最新版本
# api_key 是访问模型服务所需的认证密钥
base = lazyllm.OnlineChatModule(
    source="qwen",
    model="qwen-plus-latest",
    api_key="sk-这里填写你申请的key",
)

# 使用 with 语句创建一个意图分类器实例
# 意图分类器会根据用户输入的内容判断其意图，并分发到相应的处理模块
with IntentClassifier(base) as ic:
    # 定义聊天意图的处理方式
    # 当识别为"聊天"意图时，使用基础模型进行处理
    ic.case["聊天", base]

    # 定义画图意图的处理方式
    # 当识别为"画图"意图时，先通过基础模型将中文转换为英文绘图提示词，再调用文生图模型
    ic.case[
        "画图",
        pipeline(
            # 共享基础模型并设置绘图提示词
            base.share().prompt(
                "现在你是一位绘图提示词大师，能够将用户输入的任意中文内容转换成英文绘图提示词，在本任务中你需要将任意输入内容转换成英文绘图提示词，并且你可以丰富和扩充提示词内容。"
            ),
            # 使用通义万相文生图模型生成图像
            lazyllm.OnlineMultiModalModule(
                source="qwen",  # 使用通义千问系列模型
                model="wanx2.1-t2i-turbo",  # 使用 wanx2.1-t2i-turbo 文生图模型
                api_key="sk-这里填写你申请的key",  # API 密钥
                function="text2image",  # text2image 表示文本转图像功能
            ),
        ),
    ]

    # 定义文字转语音意图的处理方式
    # 当识别为"文字转语音"意图时，使用通义千问的文字转语音模型
    ic.case[
        "文字转语音",
        lazyllm.OnlineMultiModalModule(
            source="qwen",  # 使用通义千问模型
            model="qwen-tts",  # 使用 qwen-tts 文字转语音模型
            api_key="sk-这里填写你申请的key",  # API 密钥
            function="tts",  # tts 表示 text to speech，即文字转语音功能
        ),
    ]


# 程序入口点：启动 Web 服务
# 创建 WebModule 并启动服务
# 参数说明：
# ic: 指定处理逻辑为上面定义的意图分类器
# history=[base]: 启用对话历史记录功能，使用 base 模型管理历史
# audio=True: 启用音频功能，支持语音输入和输出
# port=23466: 指定服务运行在 23466 端口
lazyllm.WebModule(ic, history=[base], port=23466).start().wait()
