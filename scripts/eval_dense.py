import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# 1. 加载向量库
print("加载向量库...")
embeddings = np.load('../outputs/corpus_embeddings.npy')
with open('../outputs/chunk_ids.json', 'r', encoding='utf-8') as f:
    chunk_ids = json.load(f)

# 2. 加载模型 (它会直接读你刚才下好的模型，不会再联网了)
print("加载 BGE-M3 模型...")
model = SentenceTransformer('BAAI/bge-m3', device='cpu')

# ==========================================
# 3. 测试集 (直接拷过来)
# ==========================================
queries = {
    "q01": "What is the OpenAI GPT model used for?",
    "q02": "How does What 🤔 Transformers can do work?",
    "q03": "What is Whisper?"
}

qrels = {
    "q01": ["OpenAI GPT_sec_01"],
    "q02": ["What 🤔 Transformers can do_intro"],
    "q03": ["Whisper_intro","whisper_sec_01"]
}


# 4. 向量检索器
def dense_search(query, k=5):
    # 把问题变成向量
    query_embedding = model.encode(query, normalize_embeddings=True)
    # 计算余弦相似度
    scores = np.dot(embeddings, query_embedding)
    # 排序取 Top-K
    top_k_indices = np.argsort(scores)[::-1][:k]
    return [(chunk_ids[idx], scores[idx]) for idx in top_k_indices]


# 5. 评估函数
def evaluate(retriever, k=5):
    recall_sum = 0
    mrr_sum = 0
    for qid, query in queries.items():
        retrieved_chunks = [cid for cid, score in retriever(query, k)]
        relevant_chunks = set(qrels.get(qid, []))
        if not relevant_chunks: continue

        hits = sum(1 for cid in retrieved_chunks if cid in relevant_chunks)
        recall_sum += hits / len(relevant_chunks)

        for rank, cid in enumerate(retrieved_chunks, 1):
            if cid in relevant_chunks:
                mrr_sum += 1.0 / rank
                break

    return recall_sum / len(queries), mrr_sum / len(queries)


# 6. 运行
print("\n开始评估 Dense 检索...")
recall, mrr = evaluate(dense_search, k=5)

print("-" * 30)
print(f"Dense (BGE-M3) 检索结果:")
print(f"Recall@5: {recall:.4f}")
print(f"MRR:      {mrr:.4f}")
print("-" * 30)

# 7. 调试
print("\n🔍 【调试信息】Dense 实际召回的 Top-5：")
for qid, query in queries.items():
    print(f"\n问题 [{qid}]: {query}")
    print("Dense 实际找出的 Top-5:")
    results = dense_search(query, k=5)
    for rank, (cid, score) in enumerate(results, 1):
        print(f"  {rank}. ID: {cid:<25} 分数: {score:.4f}")