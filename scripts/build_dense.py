import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# 1. 加载模型（首次加载会下载权重，请确保终端已设置 HF_HOME 环境变量）
print("正在加载 BGE-M3 模型...")
model = SentenceTransformer('BAAI/bge-m3', device='cpu')
print("模型加载成功！")

# 2. 读取数据
print("读取 chunks.jsonl...")
corpus = []
chunk_ids = []
with open('../outputs/chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            chunk_ids.append(data['chunk_id'])
            corpus.append(data['text'])

# 3. 分批次生成向量（核心优化：限制 batch_size，强制 CPU 推理）
print(f"开始对 {len(corpus)} 个 chunk 生成向量...")
embeddings = []
with torch.no_grad():  # 关闭梯度计算，节省内存
    # 强制每次只送入 8 条文本，避免内存峰值过高
    batch_embeddings = model.encode(
        corpus,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=True,
        device='cpu'
    )
    embeddings = batch_embeddings

embeddings = np.array(embeddings)

# 4. 保存结果
print("正在保存向量到本地...")
np.save('../outputs/corpus_embeddings.npy', embeddings)
with open('../outputs/chunk_ids.json', 'w', encoding='utf-8') as f:
    json.dump(chunk_ids, f)

print(f"完成！向量形状: {embeddings.shape}，已保存为 corpus_embeddings.npy")