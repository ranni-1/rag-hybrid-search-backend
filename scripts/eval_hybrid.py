import json
import jieba
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

print("正在加载基础资源...")

# 1. 加载文本和 ID
corpus = []
chunk_ids = []
with open('../outputs/chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            chunk_ids.append(data['chunk_id'])
            corpus.append(data['text'])

# 2. 初始化 BM25
tokenized_corpus = [list(jieba.cut(doc.lower())) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

# 3. 加载 BGE-M3 和 向量库
embeddings = np.load('../outputs/corpus_embeddings.npy')
model = SentenceTransformer('BAAI/bge-m3', device='cpu')
print("资源加载完毕！\n")

# ==========================================
# 4. 测试集 (用调整后的 qrels)
# ==========================================
queries = {
    "q01": "What is the OpenAI GPT model used for?",
    "q02": "How does What 🤗 Transformers can do work?",
    "q03": "What is Whisper?"
}

qrels = {
    "q01": ["OpenAI GPT_sec_01"],
    "q02": ["What 🤗 Transformers can do_intro"],
    "q03": ["Whisper_intro", "Whisper_sec_01"]
}


# ==========================================
# 5. 定义两路检索引擎
# ==========================================
def bm25_search(query, k=10):  # 这里召回10个，给融合多一点候选
    tokenized_query = list(jieba.cut(query.lower()))
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunk_ids[i] for i in top_k_indices]


def dense_search(query, k=10):
    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = np.dot(embeddings, query_embedding)
    top_k_indices = np.argsort(scores)[::-1][:k]
    return [chunk_ids[i] for i in top_k_indices]


# ==========================================
# 6. RRF (倒数排名融合) 核心算法
# ==========================================
def rrf_fusion(query, k=5, rrf_k=10):
    # 1. 分别获取两路的结果 (各取前 10 名候选)
    bm25_results = bm25_search(query, k=10)
    dense_results = dense_search(query, k=10)

    rrf_scores = {}

    # 2. 计算 BM25 的 RRF 分数
    # 公式：score = 1 / (k + rank)
    for rank, cid in enumerate(bm25_results, 1):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (rrf_k + rank)

    # 3. 计算 Dense 的 RRF 分数，并累加到同一个字典里
    for rank, cid in enumerate(dense_results, 1):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (rrf_k + rank)

    # 4. 按最终得分排序，取前 k 个
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [cid for cid, score in sorted_results]


# ==========================================
# 7. 评估函数
# ==========================================
def evaluate(retriever, k=5):
    recall_sum = 0
    mrr_sum = 0
    for qid, query in queries.items():
        retrieved_chunks = retriever(query, k)
        relevant_chunks = set(qrels.get(qid, []))
        if not relevant_chunks: continue

        hits = sum(1 for cid in retrieved_chunks if cid in relevant_chunks)
        recall_sum += hits / len(relevant_chunks)

        for rank, cid in enumerate(retrieved_chunks, 1):
            if cid in relevant_chunks:
                mrr_sum += 1.0 / rank
                break

    return recall_sum / len(queries), mrr_sum / len(queries)


# ==========================================
# 8. 运行并打印
# ==========================================
print("开始评估 Hybrid (RRF) 检索...")
recall, mrr = evaluate(rrf_fusion, k=5)

print("-" * 30)
print(f"Hybrid (RRF) 融合检索结果:")
print(f"Recall@5: {recall:.4f}")
print(f"MRR:      {mrr:.4f}")
print("-" * 30)

# 看看实际融合结果
print("\n🔍 【调试信息】Hybrid 融合后的 Top-5：")
for qid, query in queries.items():
    print(f"\n问题 [{qid}]: {query}")
    print("Hybrid 找出的 Top-5:")
    results = rrf_fusion(query, k=5)
    for rank, cid in enumerate(results, 1):
        print(f"  {rank}. ID: {cid}")