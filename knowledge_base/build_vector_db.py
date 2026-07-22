import os
# 🚀 新增：强制使用 HuggingFace 国内镜像站，解决 10060 连网报错
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from sentence_transformers import SentenceTransformer

# 1. 初始化 ChromaDB 客户端 (数据将持久化保存在本地 chroma_db 文件夹中)
db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)

# 2. 创建或加载名为 medical_guidelines 的集合
collection = chroma_client.get_or_create_collection(name="medical_guidelines")

# 3. 加载本地 Embedding 模型 (通过镜像站飞速下载)
print("正在加载文本 Embedding 模型 (text2vec-base-chinese)...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 4. 模拟真实的医学指南 PDF 文档切片 (Chunking)
documents = [
    "【恶性黑色素瘤-活检原则】对于疑似黑色素瘤的病灶，首选切除活检，切缘距病灶边缘1-3mm，切忌进行切开活检或穿刺活检以免引起肿瘤播散。",
    "【恶性黑色素瘤-治疗方案】一旦确诊，需根据Breslow厚度进行前哨淋巴结活检。若发生转移，后期治疗强烈建议包含靶向治疗（如BRAF抑制剂）与免疫治疗（如PD-1单抗）。",
    "【良性角化病-临床建议】脂溢性角化病属于良性病变，通常不发生恶变。若无症状且不影响美观，无需治疗。如患者有治疗需求，可考虑液氮冷冻或激光治疗。"
]
# 为每个切片分配唯一 ID
ids = ["doc_mel_01", "doc_mel_02", "doc_bkl_01"]

print("正在将指南切片向量化并写入数据库...")
# 将文本转化为高维向量矩阵
embeddings = model.encode(documents).tolist()

# 写入数据库
collection.add(
    embeddings=embeddings,
    documents=documents,
    ids=ids
)

print(f"✅ 向量知识库构建完成！数据已持久化至: {db_path}")