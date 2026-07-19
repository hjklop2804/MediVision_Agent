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

## ✨ 核心特性

*   **真正的多模态协同:** 不是简单的图文对话，而是用硬核的医疗 CV 算法约束大模型的生成边界，彻底解决医学数值“幻觉”。
*   **极高的诊断精确度:** 底层视觉模型在 ISIC2018 数据集上，针对恶性黑色素瘤的召回率达到 **74.00%**。
*   **工业级部署标准:** 彻底的“大模型中枢”与“算法微服务”解耦设计。

