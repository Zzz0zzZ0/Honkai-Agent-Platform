# 🧠 Neuromorphic RAG Agent v2.0: LangGraph & Hybrid Search Architecture

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)
![SQLite](https://img.shields.io/badge/Database-SQLite-003B57.svg)

本项目是一个具有"类脑认知架构"的**高级混合检索与智能体决策系统**。系统摒弃了传统的单链或黑盒调用，采用有向循环图 (Cyclic Graph) 重构了大模型的推理大脑，并深度融合了强化学习与经典信息检索算法。

系统在工程实现上高度模块化，分为**感知层 (Perception)、记忆层 (Memory)、算法层 (Algorithms) 和 决策网络 (Agent)**，是一个具备极强扩展性的 Search-based LLM 应用范例。

---

## 🔥 v2.0 核心架构大升级 (v2.0 Architecture Upgrades)

在最新的 v2.0 版本中，我们针对引擎的并发稳定性、算法科学性及本地兼容性进行了史诗级重构：

1. **增量记忆与多并发安全 (Multi-tenant Memory Management)**
   - 移除了原版暴力的 ChromaDB 重置逻辑，现在支持**文档碎片增量叠加上传**，不再发生新文件覆盖旧文件的情况。
   - 引入了全局线程锁（Threading Lock）与独立的 BM25 全局内存池，彻底解决了并发状态下的底层引擎文件锁死和检索器瘫痪痛点。
2. **真实的强化学习调参闭环 (LinUCB Reward Refactoring)**
   - 废弃了基于字长和特殊字符特征的粗糙特征工程，全面改用基于 Query Embedding 切片的高维特征（8维），准确表征用户意图。
   - **打通反馈回路**：删除了自欺欺人的“余弦相似度自奖励机制”，专门对外暴露了 `/feedback` API，可以接收真实业务终端（前端页面的点赞/踩）来动态优化双路召回的最优点。
3. **SQLite 高性能舆情监控大盘 (Database Upgrade)**
   - 废除了存在巨大 I/O 阻塞隐患的 CSV 日志硬编码写入。所有高危负面评价、用户画像现在均自动归档至高可靠的 `community_feedback_log.db` 轻量级数据库中。
4. **边缘小模型的弹性容错机制 (Local Model JSON Fallback)**
   - 针对部分无法完美原生拉起 `with_structured_output` 甚至结构化输出漏参的本地模型（如 Qwen-8B 等），我们独创了底层 Regex 兜底解析机制，保障基于 LLM 的感知链路永不挂机。

---

## 🌟 核心架构与技术亮点

### 1. 前额叶皮层：基于 LangGraph 的多步决策网络 (Agent)
- **状态机流转**: 采用 `LangGraph` 替代传统的 `AgentExecutor`，将大模型的 ReAct 循环升级为透明、可控的状态机架构。
- **按需搜索路由**: Agent 能自主判断任务复杂度，在"本地知识库"与"广域网实时搜索"间精准路由与多步拆解。
- **双擎驱动**: 支持按需切换云端模型（如 Gemini 2.5 Flash）与本地开源模型。

### 2. 海马体：自适应混合检索引擎 (Hybrid RAG + LinUCB)
- **稠密与稀疏双路召回**: 底层集成 Chroma 向量库与 BM25 词频检索，兼顾语义泛化与关键词精准度。
- **LinUCB 动态权重分配**: 系统手写实现了 Contextual Bandit 算法。根据 Query Embedding，在线动态调节双路召回的 Alpha 权重，实现检索策略的自我进化。

### 3. 高阶检索算法组 (Advanced Algorithms)
- **最大边际相关性重排 (MMR)**: 引入自定义 MMR 算法对初次召回文档多样性重排，降低上下文冗余度。
- **伪相关反馈 (PRF)**: 结合 `Scikit-learn` TF-IDF，自动提取文档核心特征词，实现查询扩展，提升长尾召回率。

### 4. 边缘系统与韦尼克区：NLP 感知流水线 (Perception)
- **情感前处理 (Sentiment Analysis)**: 基于 LLM + Pydantic 结合正则兜底的 Zero-shot 结构化提取，实现多维度玩家画像（Persona）打标。
- **高危舆情入库**: 当玩家情绪为 negative 时，触发 SQLite 自动化落表以供运营后台监控分析。

### 5. 可视化交互控制台 (Streamlit UI)
- 提供完整的参数中控台，支持实时调节 Top-K、Temperature 以及各个算法模块。
- **语义空间降维可视化**: 引入 PCA 算法，实时抓取高维 Embedding 映射为 2D 散点图。

---

## 📁 模块化项目结构

```text
newcons/
├── api/
│   └── server.py           # FastAPI 后端服务 (提供 /upload, /chat, /feedback)
├── ui/
│   └── app.py              # Streamlit 可视化前端
├── core/
│   └── config.py           # 全局配置与环境变量管理
├── agent/
│   ├── graph_brain.py      # LangGraph 状态机定义
│   └── tools.py            # 工具层封装 (SQL查询、抽卡计算等)
├── engine/
│   ├── vector_store.py     # 向量存储与混合索引构建
│   └── rag_pipeline.py     # 混合检索与生成流水线 (挂载新版舆情 DB)
├── algorithms/
│   ├── linucb.py           # LinUCB v2 上下文 Bandit 算法引擎
│   ├── mmr.py              # 最大边际相关性重排
│   └── prf.py              # 伪相关反馈查询扩展
├── perception/
│   └── nlp_pipeline.py     # 基于最新 LLM JSON Fallback 的容错感知模块
└── tests/ (可选)
    ├── test_api.py         # 端到端 API 测试
    └── test_core.py        # 引擎脱库单元测试
```

---

## 🚀 快速启动指南

### 1. 基础环境
```bash
conda create -n rag_v2 python=3.10
conda activate rag_v2
pip install -r requirements.txt
```

### 2. 环境配置
配置 `.env` 文件（添加必要大模型 API 密钥）：
```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. 启动后端服务
```bash
cd newcons
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```
后端服务将在 `http://localhost:8000` 启动。

### 4. 启动前端界面（新终端）
```bash
cd newcons
streamlit run ui/app.py
```
访问 `http://localhost:8501`，即可开启全栈认知交互。
