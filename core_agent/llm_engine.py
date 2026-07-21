import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("BASE_URL"))
CV_API_URL = os.getenv("CV_MODEL_ENDPOINT")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    """工具2：本地 RAG 知识检索"""
    code = args_dict.get("disease_code", "").upper()
    db_path = os.path.join(BASE_DIR, "knowledge_base", "guidelines.json")
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            db = json.load(f)
        return db.get(code, {"error": "未找到该疾病的指南信息"})
    except Exception as e:
        return {"error": "知识库读取失败"}


# 工具路由字典 (核心机制)
AVAILABLE_TOOLS = {
    "analyze_skin_lesion": execute_analyze_skin_lesion,
    "search_clinical_guidelines": execute_search_clinical_guidelines
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

