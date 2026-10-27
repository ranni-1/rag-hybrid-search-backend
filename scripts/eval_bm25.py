import json
import jieba
from rank_bm25 import BM25Okapi

# 1. 读取我们刚才切分好的数据
print("正在加载知识库...")
corpus = []
chunk_ids = []
with open('../outputs/chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # 防止空行报错
            data = json.loads(line)
            chunk_ids.append(data['chunk_id'])
            corpus.append(data['text'])

print(f"成功加载 {len(corpus)} 个 chunk。")

# 2. 构建 BM25 索引
print("正在构建 BM25 索引...")
# 使用 jieba 对文本进行分词
tokenized_corpus = [list(jieba.cut(doc.lower())) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

# ==========================================
# 3. 准备测试集 (请务必确保这里的 ID 与 chunks.jsonl 中完全一致)
# ==========================================
queries = {
    "q01": "What is the OpenAI GPT model used for?",
    "q02": "How does What 🤗 Transformers can do work?",          # 新加的题
    "q03": "What is Whisper?"              # 新加的题
}

qrels = {
    "q01": ["OpenAI GPT_sec_10"],
    "q02": ["What 🤗 Transformers can do_intro"],                 # 换成你在 chunks.jsonl 里看到的真实 ID
    "q03": ["Whisper_sec_03"]              # 换成你看到的真实 ID
}


# 4. 检索并计算指标函数
def evaluate(retriever, k=5):
    recall_sum = 0
    mrr_sum = 0
    valid_queries = 0

    for qid, query in queries.items():
        # 获取检索结果
        retrieved_chunks = [cid for cid, score in retriever(query, k)]

        # 获取真实答案
        relevant_chunks = set(qrels.get(qid, []))
        if not relevant_chunks:
            continue  # 如果没有标注正确答案，就跳过

        valid_queries += 1

        # 计算 Recall@k
        hits = sum(1 for cid in retrieved_chunks if cid in relevant_chunks)
        recall_sum += hits / len(relevant_chunks)

        # 计算 MRR (Mean Reciprocal Rank)
        for rank, cid in enumerate(retrieved_chunks, 1):
            if cid in relevant_chunks:
                mrr_sum += 1.0 / rank
                break

    if valid_queries == 0:
        return 0, 0
    return recall_sum / valid_queries, mrr_sum / valid_queries


# 5. 定义 BM25 检索器
def bm25_search(query, k=5):
    tokenized_query = list(jieba.cut(query.lower()))
    scores = bm25.get_scores(tokenized_query)

    # 按分数从高到低排序
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunk_ids[i], scores[i]) for i in ranked_indices]


# 6. 运行评估并打印结果
print("\n开始评估...")
recall, mrr = evaluate(bm25_search, k=5)

print("-" * 30)
print(f"BM25 基线结果:")
print(f"Recall@5: {recall:.4f}")
print(f"MRR:      {mrr:.4f}")
print("-" * 30)

# 7. 【调试信息】看看 BM25 实际召回了什么，方便找错
print("\n🔍 【调试信息】看看 BM25 实际召回的 Top-5：")
for qid, query in queries.items():
    print(f"\n问题 [{qid}]: {query}")
    print(f"你标注的正确答案: {qrels.get(qid)}")
    print("BM25 实际找出的 Top-5:")

    results = bm25_search(query, k=5)
    for rank, (cid, score) in enumerate(results, 1):
        print(f"  {rank}. ID: {cid:<25} 分数: {score:.2f}")