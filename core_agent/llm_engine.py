import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

# 初始化大模型客户端 (使用 DeepSeek 兼容接口)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# 获取工具后端的地址
CV_API_URL = os.getenv("CV_MODEL_ENDPOINT")

# ==========================================
# 1. 设定 Agent 的核心人设 (System Prompt)
# ==========================================
SYSTEM_PROMPT = """
你是一个名为 MediVision 的顶尖多模态医疗辅助诊断 Agent。
你精通临床皮肤病学，并且拥有强大的计算机视觉（CV）工具辅助你进行诊断。
面对患者的询问，你必须严谨、专业。

【你的工作流】
1. 当用户提供了一张皮肤病变的图片路径时，你必须主动调用 `analyze_skin_lesion` 工具来获取诊断指标。
2. 拿到工具返回的 JSON 数据后，结合患者的文本描述，给出通俗易懂但专业的临床分析。
3. 必须在回复中体现工具返回的置信度（Confidence Score）和类别。
4. 如果工具提示“高危病灶报警”，你必须在回复中强调活检的重要性。
"""

# ==========================================
# 2. 注册工具 (Tool Schema) - 告诉大模型你有手有脚
# ==========================================
# 格式是 OpenAI/DeepSeek 官方标准的 Function Calling JSON Schema
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_skin_lesion",
            "description": "调用底层 ResNet-50 深度学习模型，对上传的皮肤镜图片进行恶性/良性分类诊断。当用户让你分析图片时必须调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "需要分析的医学图片的本地路径，例如：data/test_mel.jpg"
                    }
                },
                "required": ["image_path"]
            }
        }
    }
]


# ==========================================
# 3. 基础测试：与大模型进行一次握手
# ==========================================
def run_agent(user_input: str):
    print(f"\n👩‍⚕️ 患者: {user_input}")

    # 1. 初始化对话历史
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    print("🧠 Agent 思考中...")
    # 2. 第一次调用大模型：让它决定怎么做
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS_SCHEMA,
        temperature=0.1
    )

    response_message = response.choices[0].message

    # 3. 拦截判断：大模型是否想要调用工具？
    tool_calls = response_message.tool_calls

    if tool_calls:
        # 大模型决定使用工具！
        print(f"🛠️  Agent 决定调用工具: {tool_calls[0].function.name}")

        # 将大模型的调用指令追加到对话历史中（这是必须的上下文）
        messages.append(response_message)

        # 解析大模型提取出的参数
        function_args = json.loads(tool_calls[0].function.arguments)
        img_path = function_args.get("image_path")
        print(f"📥 提取到目标图片路径: {img_path}")

        # 4. 执行真实调用：让 Python 替大模型去请求你的 FastAPI
        api_endpoint = f"{CV_API_URL}/analyze_skin_lesion"
        print(f"🌐 正在向 CV 底层发送请求...")

        try:
            # 发送 POST 请求给你的真实模型
            api_response = requests.post(api_endpoint, json={"image_path": img_path})
            api_result = api_response.json()
            print(f"✅ CV 底层返回结果: {api_result}")
        except Exception as e:
            api_result = {"error": f"调用 CV 模型失败: {str(e)}"}
            print(f"❌ CV 底层调用失败: {str(e)}")

        # 5. 将真实的 JSON 结果转化为一条 "tool" 消息，塞回给大模型
        messages.append({
            "tool_call_id": tool_calls[0].id,
            "role": "tool",
            "name": "analyze_skin_lesion",
            "content": json.dumps(api_result, ensure_ascii=False)
        })

        print("🧠 Agent 正在结合真实数据生成最终诊断报告...")
        # 6. 第二次调用大模型：结合真实数据给出人类看得懂的报告
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3
        )
        print("\n========== 📝 最终临床分析报告 ==========")
        print(final_response.choices[0].message.content)
        print("=========================================")

    else:
        # 大模型认为不需要调用工具，直接回复
        print("\n========== Agent 直接回复 ==========")
        print(response_message.content)
        print("====================================")


if __name__ == "__main__":
    # 模拟真实用户的提问
    run_agent("医生你好，我腿上长了个黑色的斑，我非常担心，你能帮我看一下吗？图片路径是 data/ISIC_0028968.jpg")

