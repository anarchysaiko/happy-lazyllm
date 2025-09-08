# 导入 lazyllm.tools.rag 模块中的 Document 类
from lazyllm.tools.rag import Document

# 创建 Document 实例，指定数据集路径为当前目录下的 data_kb 文件夹
doc = Document(dataset_path="./data_kb")

# 使用 Document 实例的内部 reader 加载指定文件的数据
# 这里加载的是 data_kb 文件夹中的 part_1.txt 文件
data = doc._impl._reader.load_data(input_files=["./data_kb/part_1.txt"])

# 打印加载的数据内容，输出结果为data: [<Node id=2d75ea15-f278-4f78-98ba-41da7f442c81>]
print(f"data: {data}")

print(f"text: {data[0].text}")  # 这里输出结果就是 part_1.txt 的文件内容了
