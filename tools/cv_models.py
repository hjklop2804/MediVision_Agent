import os
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 0. 动态路径配置 (避免找不到文件的核心保障)
# ==========================================
# 获取当前脚本所在目录的上一级目录（即 MediVision_Agent 根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

app = FastAPI(title="MediVision CV Tools", description="Agent 的计算机视觉底层接口")


class ImageRequest(BaseModel):
    image_path: str


# ==========================================
# 1. 还原真实网络骨架并加载权重
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"========== 系统初始化 ==========")
print(f"当前推理设备: {DEVICE}")

# 构建基础 ResNet-50，修改为 7 分类
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)

# 加载消融实验中最强的模型权重
weight_path = os.path.join(ASSETS_DIR, "ResNet50_FocalLoss_Augmented_best.pth")
if os.path.exists(weight_path):
    # 使用 weights_only=True 提高安全性（PyTorch 新版推荐做法）
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    print(f"✅ ISIC2018 模型权重加载成功！({weight_path})")
else:
    print(f"❌ 警告：找不到权重文件 {weight_path}")

model.to(DEVICE)
model.eval()

# ==========================================
# 2. 真实图像预处理流与标签映射
# ==========================================
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = {
    0: 'MEL (恶性黑色素瘤)',
    1: 'NV (黑色素细胞痣)',
    2: 'BCC (基底细胞癌)',
    3: 'AKIEC (光化性角化病)',
    4: 'BKL (良性角化病)',
    5: 'DF (皮肤纤维瘤)',
    6: 'VASC (血管病变)'
}


# ==========================================
# 3. 核心推理接口
# ==========================================
@app.post("/analyze_skin_lesion")
async def analyze_skin_lesion(request: ImageRequest):
    # 如果传来的是相对路径，将其转换为绝对路径
    target_img_path = request.image_path
    if not os.path.isabs(target_img_path):
        target_img_path = os.path.join(BASE_DIR, target_img_path)

    if not os.path.exists(target_img_path):
        raise HTTPException(status_code=404, detail=f"找不到该图像文件: {target_img_path}")

    try:
        # 1. 读取并处理真实图像
        image = Image.open(target_img_path).convert('RGB')
        tensor_image = data_transforms(image).unsqueeze(0).to(DEVICE)

        # 2. 执行模型前向推理
        with torch.no_grad():
            outputs = model(tensor_image)
            probs = torch.softmax(outputs, dim=1)[0]

            # 获取预测的类别索引和对应的置信度
            confidence_score, predicted_idx = torch.max(probs, 0)
            pred_class_id = predicted_idx.item()
            pred_class_name = CLASS_NAMES[pred_class_id]
            conf_val = round(confidence_score.item(), 4)

        # 3. 生成给 Agent 大脑的结构化报告
        warning_msg = "未见明显高危特征。"
        if pred_class_id in [0, 2]:  # MEL 和 BCC 属于恶性/高危
            warning_msg = "🚨 高危病灶报警：该病变具有高度恶性特征，建议立刻结合皮肤镜图谱与临床指南进行复核并安排活检。"

        return {
            "status": "success",
            "task": "Skin Lesion Classification",
            "prediction": pred_class_name,
            "confidence_score": conf_val,
            "heatmap_generated": False,
            "clinical_warning": warning_msg
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("🚀 正在启动 MediVision CV API 服务...")
    uvicorn.run(app, host="127.0.0.1", port=8000)