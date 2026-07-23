import os
import re

# ==========================================
# 🛑 删掉所有花里胡哨的 SSL 魔改，只保留这句在之前成功跑通的镜像配置！
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("BASE_URL"))
CV_API_URL = os.getenv("CV_MODEL_ENDPOINT")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 🚀 新增：全局初始化向量数据库与 Embedding 模型
# ==========================================
print("正在加载向量检索中枢...")
embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
db_path = os.path.join(BASE_DIR, "knowledge_base", "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(name="medical_guidelines")


# ==========================================
# 1. 进阶版 System Prompt (强调查阅指南)
# ==========================================
SYSTEM_PROMPT = """
你是一个名为 MediVision 的顶尖多模态医疗辅助诊断 Agent。
你拥有两个核心工具：
1. `analyze_skin_lesion`: 用于查看患者的皮肤镜图片并得出初步分类和置信度。
2. `search_clinical_guidelines`: 用于检索该类疾病的最新临床治疗指南。

【严格的 ReAct 工作流】
1. 收到图片后，必须先调用 `analyze_skin_lesion` 获取 AI 视觉诊断结果。
2. 获取到视觉诊断结果后，如果是恶性肿瘤（如 MEL），你**必须**接着调用 `search_clinical_guidelines` 工具，传入疾病缩写（如 "MEL"），检索权威指南。
3. 收集齐视觉证据和权威指南后，使用 <thinking> 标签进行内部逻辑梳理。
4. 最终输出 Markdown 报告，必须包含：AI分类及置信度、热力图展示、以及**严格引用指南原文**的临床建议。
"""

# ==========================================
# 2. 注册双工具 Schema
# ==========================================
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_skin_lesion",
            "description": "调用 CV 模型分析皮肤镜图片",
            "parameters": {
                "type": "object",
                "properties": {"image_path": {"type": "string"}},
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_brain_mri",
            "description": "调用 3D 医疗影像分析模型分析脑部核磁共振(MRI)数据。当患者提及脑部、头痛或提供 .nii.gz 格式文件时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {"mri_path": {"type": "string"}},
                "required": ["mri_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_clinical_guidelines",
            "description": "检索特定皮肤病的临床诊疗指南",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease_code": {
                        "type": "string",
                        "description": "疾病的缩写代码，例如：MEL, BKL"
                    }
                },
                "required": ["disease_code"]
            }
        }
    }
]


# ==========================================
# 3. 本地工具函数实现
# ==========================================
def execute_analyze_skin_lesion(args_dict):
    """工具1：请求底层视觉 FastAPI"""
    img_path = args_dict.get("image_path")
    try:
        response = requests.post(f"{CV_API_URL}/analyze_skin_lesion", json={"image_path": img_path})
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def execute_search_clinical_guidelines(args_dict):
    """工具2：基于 ChromaDB 的向量语义检索"""
    # 提取大模型想要查询的关键词 (例如: "MEL", "恶性黑色素瘤")
    query = args_dict.get("disease_code", "")
    if not query:
        return {"error": "检索词为空"}

    try:
        print(f"🔍 正在向量库中检索相关临床文献，关键词: {query}")
        # 1. 将查询词转化为向量
        query_embedding = embedding_model.encode([query]).tolist()

        # 2. 在 ChromaDB 中检索余弦相似度最高的 2 个文本块 (Top-K = 2)
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=2
        )

        # 3. 提取并拼接检索到的文本
        if results and results['documents'] and len(results['documents'][0]) > 0:
            retrieved_texts = results['documents'][0]
            combined_guidelines = "\n".join(retrieved_texts)
            return {"status": "success", "retrieved_guidelines": combined_guidelines}
        else:
            return {"error": "未检索到相关指南信息"}

    except Exception as e:
        return {"error": f"向量检索失败: {str(e)}"}

def execute_analyze_brain_mri(args_dict):
    """工具3：3D 脑肿瘤分析 (Mock 占位)"""
    return {
        "status": "pending",
        "message": "🧠 脑肿瘤 3D 分割专科模型目前正在集群进行最后阶段的训练迭代，预计近期上线。当前系统暂时建议患者以线下专业医生的影像学诊断为准。"
    }

# 工具路由字典 (核心机制)
AVAILABLE_TOOLS = {
    "analyze_skin_lesion": execute_analyze_skin_lesion,
    "search_clinical_guidelines": execute_search_clinical_guidelines
}

AVAILABLE_TOOLS = {
    "analyze_skin_lesion": execute_analyze_skin_lesion,
    "search_clinical_guidelines": execute_search_clinical_guidelines,
    "analyze_brain_mri": execute_analyze_brain_mri  # 新增占位工具
}

# ==========================================
# 4. 完整的 ReAct 智能循环
# ==========================================
def run_agent(user_input: str):
    print(f"\n👩‍⚕️ 患者: {user_input}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_input}]

    # 开启最大 3 轮的思维循环，允许 Agent 连续调用多个工具
    for step in range(3):
        print(f"\n🧠 Agent 思考中 (第 {step + 1} 轮)...")
        response = client.chat.completions.create(
            model="deepseek-chat", messages=messages, tools=TOOLS_SCHEMA, temperature=0.1
        )
        response_msg = response.choices[0].message

        # 如果模型不想用工具了，说明思考结束，直接输出最终回答
        if not response_msg.tool_calls:
            print("\n========== 📝 最终临床分析报告 ==========")
            print(response_msg.content)
            print("=========================================")
            break

        # 如果模型想用工具，将其意图加入上下文
        messages.append(response_msg)

        # 遍历执行大模型想要调用的所有工具
        for tool_call in response_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"🛠️  触发动作: {func_name} | 参数: {func_args}")

            # 通过路由字典动态调用 Python 函数
            function_to_call = AVAILABLE_TOOLS.get(func_name)
            if function_to_call:
                result = function_to_call(func_args)
                print(f"✅ 工具返回结果: {result}")
            else:
                result = {"error": "Tool not found"}

            # 将工具执行结果作为 context 塞回给大模型
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": func_name,
                "content": json.dumps(result, ensure_ascii=False)
            })


if __name__ == "__main__":
    # 找一张恶性黑色素瘤的照片测试
    run_agent("医生你好，我腿上长了个黑色的斑，你能帮我看一下吗？图片路径是 data/ISIC_0025368.jpg")


# ==========================================
# 🚀 Web UI 专属调用接口
# ==========================================
def run_agent_for_ui(image_path):
    """供 Web 端调用的 Agent 函数，返回 (最终报告, 热力图路径)"""
    user_input = f"医生你好，你能帮我看一下这个皮损图片吗？图片路径是 {image_path}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_input}]

    final_report = ""
    heatmap_path = None

    def strip_thinking_blocks(text: str) -> str:
        """去除 <thinking>...</thinking> 内心独白，避免破坏 Markdown 渲染，也避免暴露给用户"""
        if not text:
            return text
        return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
    # 扩大思考上限，防止过早截断
    for step in range(5):
        print(f"🧠 [Agent 思考中 - 第 {step + 1} 轮] 正在与大模型中枢通讯，请稍候...")

        response = client.chat.completions.create(
            model="deepseek-chat", messages=messages, tools=TOOLS_SCHEMA, temperature=0.1
        )
        response_msg = response.choices[0].message

        # 如果大模型不再调用工具，说明它开始撰写最终报告了！
        if not response_msg.tool_calls:
            print("📝 [Agent 思考完毕] 已经拿到大模型回复！")
            final_report = response_msg.content
            final_report = strip_thinking_blocks(final_report)
            # 👇 新增下面这几行，强制把它在后台写好的报告打印出来！
            print("\n" + "=" * 20 + " 诊断报告预览 " + "=" * 20)
            print(final_report)
            print("=" * 54 + "\n")
            print("🚀 正在将结果推送给网页前端，如果网页依然转圈，请立刻关闭电脑的代理软件！")
            break

        messages.append(response_msg)

        # 执行工具
        for tool_call in response_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            function_to_call = AVAILABLE_TOOLS.get(func_name)
            if function_to_call:
                result = function_to_call(func_args)
                if func_name == "analyze_skin_lesion" and "heatmap_path" in result:
                    heatmap_path = result["heatmap_path"]
            else:
                result = {"error": "Tool not found"}

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": func_name,
                "content": json.dumps(result, ensure_ascii=False)
            })

    print(f"🔎 final_report 类型: {type(final_report)}, 内容repr: {final_report!r}")
    print(f"🔎 heatmap_path 类型: {type(heatmap_path)}, 内容repr: {heatmap_path!r}")
    return final_report, heatmap_path