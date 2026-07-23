import os
import tempfile
import traceback
# ==========================================
# 🛑 绝杀 1：物理级拔除所有代理环境变量，防止本地数据包被拐跑
# ==========================================
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["all_proxy"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
from core_agent.llm_engine import run_agent_for_ui


def process_case(img_filepath):
    if not img_filepath:
        yield "⚠️ 请先上传病灶图片！", None
        return

    yield "🧠 Agent 正在思考中，请稍候...", None

    try:
        report, heatmap_path = run_agent_for_ui(img_filepath)
    except Exception as e:
        traceback.print_exc()
        yield f"❌ Agent 执行异常: {e}", None
        return

    yield report, None  # 先把报告推上去，保证一定显示

    if heatmap_path and os.path.exists(heatmap_path):
        yield report, heatmap_path  # 图片单独再更新一次


# ==========================================
# 🎨 构筑 UI 界面
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 MediVision-Agent: 多模态医疗辅助诊断中枢")
    gr.Markdown(
        "上传皮肤镜图像，Agent 将自动调度底层 **CV 视觉模型** 与 **RAG 临床指南知识库**，为您生成带有热力图溯源的专家级诊断报告。")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="filepath", label="📸 上传病灶图像 (JPG/PNG)")
            submit_btn = gr.Button("🚀 启动 Agent 智能诊断", variant="primary")

            gr.Markdown("### 🔬 视觉特征溯源")
            heatmap_output = gr.Image(type="filepath", label="AI 视觉关注热力图 (Grad-CAM)")

        with gr.Column(scale=2):
            gr.Markdown("### 📝 Agent 综合分析报告")
            report_output = gr.Markdown(
                value="等待诊断中...\n（注：包含检索文献及推理过程，整个闭环可能需要 15-20 秒，请耐心等待。）")

    submit_btn.click(
        fn=process_case,
        inputs=[img_input],
        outputs=[report_output, heatmap_output]
    )

if __name__ == "__main__":
    print("🌐 正在启动 MediVision Web UI...")
    # 🛑 绝杀 2：开启 allowed_paths 豁免权！
    # 强制允许前端读取 C 盘临时目录和 E 盘项目目录的文件，彻底粉碎 Gradio 4.0+ 的 403 拦截！
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=["C:\\", "D:\\", "E:\\", tempfile.gettempdir()]
    )