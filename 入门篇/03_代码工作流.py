import lazyllm
from lazyllm import pipeline

# Part1: 聊天机器人模块 - 用于优化提示词
prompt_enhancer = lazyllm.OnlineChatModule(
    source="qwen",
    model="qwen-plus-latest",  # 使用一个标准的模型名称
    api_key="sk-这里填写你申请的key",
)

# 为提示词优化器添加提示词模板
enhancer_prompt = """You are a drawing prompt word master who can convert any Chinese content entered by the user into English drawing prompt words. In this task, you need to convert any input content into detailed and rich English drawing prompt words that can be used for AI art generation.

Please follow these guidelines:
1. Translate the Chinese input into English
2. Expand the prompt with rich descriptive details
3. Include artistic style, lighting, colors, and other relevant elements
4. Structure the response clearly for the next processing step

Example:
Input: 画一只小猪
Output: A cute little pig, chubby and adorable with a round pink body, curly tail, small pointed ears, and a snout with two nostrils. The pig has sparkling eyes with long eyelashes, standing on four short legs with tiny hooves. Soft pastel pink color palette, friendly expression, cartoon-style illustration, clean simple background"""

# 提示词优化器使用上面的提示词模版完成任务
prompt_enhancer_with_template = prompt_enhancer.prompt(
    lazyllm.ChatPrompter(enhancer_prompt)
)

# Part2: 文生图模块 - 使用通义万象模型进行图像生成
text_to_image_module = lazyllm.OnlineMultiModalModule(
    source="qwen",
    model="wanx2.1-t2i-turbo",
    api_key="sk-这里填写你申请的key",
    function="text2image",
)

# 为文生图模型添加特定提示词，明确要求生成图像
image_generation_prompt = """Based on the following detailed prompt, please generate an image:

{input}

Please create a high-quality image that matches this description exactly.
Include all the details mentioned in the prompt, such as colors, style, lighting, and composition.
Make sure the image is visually appealing and accurately represents the description provided."""

# 文生图模块使用上面的文生图提示词完成任务
text_to_image_with_template = text_to_image_module.prompt(
    lazyllm.ChatPrompter(image_generation_prompt)
)

# 创建 pipeline
with pipeline() as ppl:
    ppl.enhancer = prompt_enhancer_with_template
    ppl.t2i = text_to_image_with_template

# 创建 WebModule并启动服务
web = (
    lazyllm.WebModule(
        ppl,
        port=23466,
        history=[ppl.enhancer],  # 只为提示词优化器保留历史记录
    )
    .start()
    .wait()
)
