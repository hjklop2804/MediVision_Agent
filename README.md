
# 🩺 MediVision-Agent: 多模态医疗辅助诊断智能体

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)
![DeepSeek API](https://img.shields.io/badge/LLM-DeepSeek-4d6bfe.svg)

## 📖 项目简介
MediVision 是一款结合了**大语言模型 (LLM)** 与**底层深度学习视觉模型 (CV)** 的多模态医疗辅助诊断 Agent。本项目旨在解决传统 AI 医疗模型“黑盒化”和“缺乏交互”的痛点，通过 Agent 的 ReAct 范式，赋予大模型调用专业医疗算法的能力，从而生成具备高置信度和临床解释性的结构化诊断报告。

当前版本已打通 **ISIC2018 皮肤镜图像**的智能分析闭环，未来将扩展至脑肿瘤 MRI 等多模态分析场景。

## 🧠 核心架构设计 (Agentic Workflow)

本项目摒弃了臃肿的 LangChain 框架，采用纯原生 Python 手写底层的核心运转逻辑，以确保对系统拥有 100% 的掌控力与极低的响应延迟。

*   **大脑层 (Cognitive Engine):** 基于大模型 API，系统被赋予了资深皮肤科专家的 System Prompt。利用 **ReAct (Reason + Act)** 框架，大模型能够根据用户的自然语言输入，自主决定是否需要调用视觉分析工具，并解析视觉工具返回的数据。
*   **工具层 (Tool Use):** 将传统的深度学习推理脚本当作 Agent 的“手脚”。通过 FastAPI 将基于 PyTorch 构建的 **ResNet-50 (重构 Focal Loss, 解决临床数据长尾分布)** 封装为标准微服务接口。
*   **协同机制:** 大模型输出 `tool_calls` 指令拦截 -> 发起本地 HTTP 请求 -> 核心算法进行 Tensor 推理运算 -> 返回 JSON 格式的置信度与类别 -> 大模型进行二次结合推理与报告生成。
*   **可视化呈现 (Web UI):** 基于 `Gradio` 构建全双工通信界面，支持实时流式打印 Agent 内部的思维链（CoT），并同步对比原图与 Grad-CAM 高危病灶热力图。
## ✨ 核心工程亮点

*   **多模态协同与精准拦截:** 不是简单的图文对话，而是用硬核的医疗 CV 算法约束大模型的生成边界，彻底解决医学数值“幻觉”。
*   **极高的诊断精确度:** 底层视觉模型在 ISIC2018 数据集上，针对恶性黑色素瘤的召回率达到 **74.00%**。
*   **工业级部署标准:** 彻底的“大模型中枢”与“算法微服务”解耦设计。
*   **全栈排障实战：突破前端 XSS 渲染Bug:** 在打通全链路的最终阶段，系统曾遇到“后端数据完美生成，前端网页静默崩溃转圈”的极限暗坑。

    根因溯源： 为提升推理透明度，System Prompt 强制大模型输出 <thinking>...</thinking> 思维链标签。但 Gradio 前端在渲染 Markdown 时，底层 rehype-sanitize 净化器将其判定为未知危险标签，触发了激进的 XSS 防御机制，导致合法的 DOM 树结构整体坍塌，页面后续元素被静默吞噬（后端无任何异常报错）。

    解决办法： 在 Python 后端下发数据前，引入正则清洗中间件，将伪标签动态转换为合法的 Markdown 代码块。这一方案不仅秒解假死问题，还顺势在前端界面上方漂亮地展示了 Agent 的底层推理逻辑。
## 🚀 快速启动

### 1. 环境准备
```bash
# 从零开始配置独立环境
conda create -n med_agent python=3.10 -y
conda activate med_agent
pip install -r requirements.txt

```

### 2. 配置密钥

在根目录创建 .env 文件并填入配置：
```bash
env
DEEPSEEK_API_KEY="sk-your-api-key"
BASE_URL="[https://api.deepseek.com/v1](https://api.deepseek.com/v1)"
CV_MODEL_ENDPOINT="[http://127.0.0.1:8000](http://127.0.0.1:8000)"

```

### 3. 双端运行

为了模拟真实的微服务架构，请打开两个终端窗口：

**终端 A (启动视觉算法底座):**

```bash
python tools/cv_models.py

```

*服务将在 `localhost:8000` 启动，并自动加载本地 `.pth` 权重。*

**终端 B (唤醒 Agent 大脑):**

```bash
python web_ui.py

```

## 🛠️ Prompt 调优与 Bad Case 分析

在多模态数据（文本+视觉 JSON）融合的早期测试中，系统偶尔会出现“无视低置信度”或“遗漏临床警告”的幻觉（Hallucination）。

本项目通过在 Agent 中枢引入严格的 **ReAct 约束** 和 **内部思维链 (CoT)**，并结合 **Few-Shot (少样本提示)**，彻底解决了大模型在医疗指标上的漏读问题，成功实现了对高危病灶的精准拦截与急救引导。

👉 **[点击这里查看完整的 Prompt 调优与 Bad Case 分析记录（附真实高危 Case 拦截截图）](docs/Prompt_Tuning_and_BadCase_Analysis.md)**

## 📅 演进路线图 (To-Do)

* [x] 完成 ResNet-50 FastAPI 工具封装与 7 分类 JSON 结构化输出。
* [x] 完成 LLM 的 ReAct 循环机制搭建，实现自动化 Tool Use。
* [x] **集成 Grad-CAM:** 让工具在返回数值的同时，生成病灶关注热力图，提升视觉可解释性。
* [x] **引入 RAG 知识检索:** 挂载最新版《中国黑色素瘤诊疗指南》，强制大模型在诊断时进行规范化引用。
* [x] **全双工交互:** 完成 Gradio Web UI 界面开发与 XSS 渲染对抗处理。
* [ ] **多工具路由:** 接入 BraTS2019 3D U-Net 脑肿瘤分析工具，测试大模型在面对不同模态数据时的 Tool 路由准确率。

## 👤 开发者



* GitHub: [@hjklop2804](https://github.com/hjklop2804)



