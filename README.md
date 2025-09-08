# happy-lazyllm

happy-lazyllm-tutorial 配套代码环境，全程基于 uv 配置。

## 项目介绍

本项目是 happy-lazyllm 教程的配套代码，旨在帮助用户快速上手并实践大语言模型相关的技术。项目涵盖了从基础的聊天机器人构建到高级的 RAG（Retrieval-Augmented Generation）技术实现。

## 目录结构

- `入门篇/`: 包含构建第一个聊天机器人、代码工作流、多模态聊天等基础教程代码
- `进阶篇/`: 包含 RAG 基础概念、文档解析器、检索器和召回、实战召回优化、相似度优化、重排序与多路召回等高级教程代码
- `data_kb/`: 知识库数据文件
- `datasets/`: 数据集文件
- `rag_data/`: RAG 相关数据文件

## 安装步骤

1. 确保已安装 [uv](https://github.com/astral-sh/uv)
2. 克隆项目代码：
   ```bash
   git clone https://github.com/anarchysaiko/happy-lazyllm
   ```
3. 进入项目目录：
   ```bash
   cd happy-lazyllm
   ```
4. 安装依赖：
   ```bash
   uv sync
   ```

## 使用方法

- 运行入门篇教程代码：

  ```bash
  uv run 入门篇/02_构建你的第一个聊天机器人.py
  ```

- 运行进阶篇教程代码：
  ```bash
  uv run 进阶篇/01_RAG的基础概念和实现.py
  ```

## 许可证

本项目采用 MIT 许可证，详情请见 [LICENSE](LICENSE) 文件。
