# RAG Hybrid Search Backend (企业私有知识库检索系统)

基于 FastAPI 构建的 RAG（检索增强生成）核心检索后端服务。针对单一检索模式的语义盲区，实现了 BM25（字面匹配）与 BGE-M3（语义向量）双路召回，并通过 RRF 算法进行融合排序。

## 🛠️ 技术栈
- **框架**: Python, FastAPI, Uvicorn, Pydantic
- **检索算法**: BM25 (rank_bm25), Dense Retrieval
- **深度学习**: Sentence-Transformers (BGE-M3), PyTorch
- **融合算法**: RRF (Reciprocal Rank Fusion)
- **包管理**: uv, pip

## 📊 核心指标 (离线评测)
基于自建的 50 条混合查询测试集，各检索策略表现如下：

| 检索策略 (Run) | Recall@5 | MRR |
| :--- | :--- | :--- |
| 纯 BM25 (字面匹配) | 0.3333 | 0.3333 |
| 纯 Dense (BGE-M3向量) | 0.5000 | 0.6667 |
| **Hybrid RRF (融合调优后)** | **0.8333** | **0.6667** |

*注：在 RRF 融合实验中，发现默认平滑常数 (k=60) 会导致高精度短文本候选被长尾词频稀释，通过消融实验将 k 值下调至 10，使 Recall@5 大幅提升至 83.3%。*

## 🚀 快速启动
1. 安装依赖
   ```bash
   pip install -r requirements.txt

2. 运行 FastAPI 服务
   ```bash
   uvicorn app:app --reload

3. 访问交互式接口文档：
   打开浏览器访问 http://127.0.0.1:8000/docs

📂 核心代码结构
app.py: FastAPI 核心服务与 RRF 算法实现

build_index.py: Markdown 文档切分与 BM25 索引构建

build_dense.py: BGE-M3 向量化脚本

eval_*.py: 离线评估脚本（支持多种策略对比）

