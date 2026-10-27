import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线，不要联网检查更新
from fastapi import FastAPI
from pydantic import BaseModel
import json
import jieba
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 初始化 FastAPI
# ==========================================
app = FastAPI(title="RAG Hybrid Search API")

# ==========================================
# 2. 全局加载资源 (启动时只加载一次，不会每次都卡)
# ==========================================
print("正在加载基础资源...")
corpus = []
chunk_ids = []
with open('outputs/chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            chunk_ids.append(data['chunk_id'])
            corpus.append(data['text'])

# 建立 ID 到 文本的映射字典，方便接口返回
id_to_text = dict(zip(chunk_ids, corpus))

tokenized_corpus = [list(jieba.cut(doc.lower())) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

embeddings = np.load('outputs/corpus_embeddings.npy')
# 注意：为了防止联网卡住，这里强制加载本地
model = SentenceTransformer('BAAI/bge-m3', device='cpu',local_files_only=True)


# ==========================================
# 3. 定义检索与 RRF 融合逻辑
# ==========================================
def bm25_search(query, k=10):
    tokenized_query = list(jieba.cut(query.lower()))
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunk_ids[i] for i in top_k_indices]


def dense_search(query, k=10):
    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = np.dot(embeddings, query_embedding)
    top_k_indices = np.argsort(scores)[::-1][:k]
    return [chunk_ids[i] for i in top_k_indices]


def rrf_fusion(query, k=3, rrf_k=10):
    bm25_results = bm25_search(query, k=10)
    dense_results = dense_search(query, k=10)
    rrf_scores = {}

    for rank, cid in enumerate(bm25_results, 1):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(dense_results, 1):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (rrf_k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [cid for cid, score in sorted_results]


# ==========================================
# 4. 定义 API 请求体格式
# ==========================================
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


# ==========================================
# 5. 定义 API 路由
# ==========================================
@app.post("/search")
def search_api(req: SearchRequest):
    # 调用 RRF 融合获取 Top-K 的 chunk_id
    results = rrf_fusion(req.query, k=req.top_k)

    # 拼装返回结构，把纯文本也带上，方便前端或大模型直接读取
    formatted_results = []
    for cid in results:
        formatted_results.append({
            "chunk_id": cid,
            "content": id_to_text.get(cid, "未找到文本")
        })

    return {
        "code": 200,
        "query": req.query,
        "results": formatted_results
    }