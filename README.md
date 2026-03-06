# 🧠 Neuromorphic RAG Agent: LangGraph & Hybrid Search Architecture

本项目是一个具有"类脑认知架构"的**高级混合检索与智能体决策系统**。系统摒弃了传统的单链或黑盒调用,采用有向循环图 (Cyclic Graph) 重构了大模型的推理大脑,并深度融合了强化学习与经典信息检索算法。

系统在工程实现上高度模块化,分为**感知层 (Perception)、记忆层 (Memory)、算法层 (Algorithms) 和 决策网络 (Agent)**,是一个具备极强扩展性的 Search-based LLM 应用范例。

## 🌟 核心架构与技术亮点

### 1. 前额叶皮层：基于 LangGraph 的多步决策网络 (Agent)
* **状态机流转**: 采用 `LangGraph` 替代传统的 `AgentExecutor`,将大模型的 ReAct (Reason+Act) 循环升级为高度透明、可控的状态机架构。
* **按需搜索路由**: Agent 能够自主判断任务复杂度,在"本地私有知识库 (Local Knowledge)"与"广域网实时搜索 (Web Search)"之间进行精准路由与多步拆解。
* **双擎驱动**: 支持按需切换云端模型（如 Gemini 2.5 Flash）与本地开源模型（如 Qwen 3）。

### 2. 海马体：自适应混合检索引擎 (Hybrid RAG + LinUCB)
* **稠密与稀疏双路召回**: 底层集成 Chroma 向量库与 BM25 词频检索,兼顾语义泛化与关键词精准度。
* **LinUCB 动态权重分配 (自适应 Alpha)**: 区别于传统的固定权重,系统手写实现了 Contextual Bandit (上下文多臂老虎机) 算法。根据 Query 的长度、特殊字符密度等特征,**在线动态调节**双路召回的 Alpha 权重,实现检索策略的自我进化。

### 3. 高阶检索算法组 (Advanced Algorithms)
* **最大边际相关性重排 (MMR)**: 引入自定义的 MMR 算法对初次召回文档进行多样性重排,有效降低上下文冗余度。
* **伪相关反馈 (PRF)**: 结合 `Scikit-learn` 的 TF-IDF 算法,自动提取初次召回文档的核心特征词,实现查询发散与扩展 (Query Expansion),大幅提升长尾问题的召回率。

### 4. 边缘系统与韦尼克区：NLP 感知流水线 (Perception)
* **情感前处理 (Sentiment Analysis)**: 集成轻量级 RoBERTa 模型,捕捉用户输入的潜在情绪（如焦虑/积极）,并动态注入到大模型的 System Prompt 中,实现"带温度"的生成。
* **实体识别后处理 (NER)**: 提取大模型生成结果中的关键实体（人名、地名、机构等）,在前端 UI 进行可视化增强。

### 5. 可视化交互控制台 (Streamlit UI)
* 提供完整的参数中控台,支持实时调节 Top-K、Temperature 以及各个算法模块的开关。
* **语义空间降维可视化**: 引入 PCA 算法,实时抓取 ChromaDB 中的高维 Embedding 向量,映射为 2D 散点图,直观展示知识库的数据分布。

---

## 📁 模块化项目结构

```text
├── app.py                  # Streamlit 可视化前端与应用入口
├── core/
│   └── config.py           # 全局配置与环境变量管理 (Embed模型、代理设置等)
├── agent/
│   ├── graph_brain.py      # LangGraph 状态机定义与节点编排
│   └── tools.py            # 工具层封装,串联底层 RAG 与算法模块
├── memory/
│   └── rag_engine.py       # 混合索引构建、Chroma/BM25 初始化及 PCA 降维
├── algorithms/
│   ├── linucb.py           # LinUCB 上下文 Bandit 算法引擎
│   ├── mmr.py              # 最大边际相关性重排算法
│   └── prf.py              # 伪相关反馈查询扩展算法
└── perception/
    └── nlp_pipeline.py     # 基于 HuggingFace Pipeline 的情感与 NER 模块
```

---

## 🚀 快速启动指南

### 1. 基础环境

推荐使用 Python 3.10+,克隆代码并安装依赖：

```bash
git clone https://github.com/您的用户名/RAG-Agent.git
cd RAG-Agent
pip install -r requirements.txt
```

### 2. 环境配置

在项目根目录创建 `.env` 文件,配置必要的大模型 API 密钥（切勿将此文件提交至 Git）：

```env
GOOGLE_API_KEY=your_api_key_here
```

(注：如需在国内网络环境使用 API,请在 `core/config.py` 中确认代理端口配置。)

### 3. 运行系统

```bash
streamlit run app.py
```

打开浏览器访问,在左侧边栏上传 PDF 或 TXT 文档以"注入记忆",即可开启全栈认知交互！
