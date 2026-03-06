import csv
import os
from datetime import datetime

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as lc_tool  # if needed elsewhere
from pydantic import BaseModel, Field

from engine.vector_store import embeddings
from perception.nlp_pipeline import analyze_user_query
from algorithms.linucb import linucb_agent
from algorithms.prf import algo_pseudo_relevance_feedback
from algorithms.mmr import algo_mmr_rerank


# ==========================================
# 核心单步检索回路 (包含舆情落表)
# ==========================================


def get_answer_complex(
    vectorstore,
    bm25_retriever,
    question,
    k_param: int = 3,
    temp_param: float = 0.1,
    alpha: float = 0.5,
    model_type: str = "cloud",
    use_multiquery: bool = False,
    use_rerank: bool = False,
    use_auto_alpha: bool = False,
    use_emotion: bool = False,
    use_ner: bool = False,
):
    emotion_label = "neutral"
    extracted_entities = []
    player_persona = []

    # 0. NLP 结构化感知 (情绪、实体、玩家画像)
    if use_emotion or use_ner:
        emotion_label, extracted_entities, player_persona = analyze_user_query(
            question, model_type=model_type, temp_val=temp_param
        )

        # 舆情自动化监控：连同玩家画像落表
        if emotion_label == "negative":
            log_file = "community_feedback_log.csv"
            file_exists = os.path.isfile(log_file)
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        ["Timestamp", "Player_Query", "Emotion", "Player_Persona", "Status"]
                    )
                persona_str = "|".join(player_persona) if player_persona else "未知"
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        question,
                        emotion_label,
                        persona_str,
                        "Pending_Review",
                    ]
                )

    # 1. 模型初始化
    if model_type == "local":
        llm = ChatOllama(model="qwen3:8b", temperature=temp_param)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp_param)

    # 2. LinUCB
    final_alpha = alpha
    arm_idx = -1
    context_vec = None
    if use_auto_alpha:
        arm_idx, final_alpha, context_vec = linucb_agent.select_arm(question)

    # 3. 混合检索 (增加空载保护)
    initial_docs = []
    seen_contents = set()
    
    # 【修复点 1】：只有在记忆库存在时，才启动检索
    if vectorstore is not None and bm25_retriever is not None:
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        # 分别独立召回
        docs_vs = chroma_retriever.invoke(question)
        docs_bm25 = bm25_retriever.invoke(question)

        # 根据 LinUCB 动态权重 (final_alpha) 精准分配配额
        k_vs = max(1, int(15 * final_alpha))
        k_bm25 = max(1, int(15 * (1 - final_alpha)))

        # 手动合并与去重
        for doc in (docs_vs[:k_vs] + docs_bm25[:k_bm25]):
            if doc.page_content not in seen_contents:
                initial_docs.append(doc)
                seen_contents.add(doc.page_content)

    # 4. PRF 发散思维
    search_queries = [question]
    if use_multiquery and initial_docs: # 【修复点 2】：有初始文档才发散
        search_queries = algo_pseudo_relevance_feedback(question, initial_docs)
        if len(search_queries) > 1:
            pass
    unique_docs = list({doc.page_content: doc for doc in initial_docs}.values())

    # 5. LinUCB 奖励更新 (增加保护，避免没文档时去算相似度)
    if use_auto_alpha and unique_docs: # 【修复点 3】
        q_vec = embeddings.embed_query(question)
        d_vec = embeddings.embed_query(unique_docs[0].page_content)
        reward = cosine_similarity([q_vec], [d_vec])[0][0]
        linucb_agent.update(arm_idx, context_vec, reward)

    # 6. MMR 重排序
    final_docs = (
        algo_mmr_rerank(question, unique_docs, embeddings, k_param=k_param)
        if use_rerank and unique_docs
        else unique_docs[:k_param]
    )

    # 7. 生成阶段
    tone_instruction = (
        "检测到用户情绪焦虑。请使用安抚性语气回答。" if emotion_label == "negative" else ""
    )
    
    # 如果没有文档，就不要强行插入“【记忆片段】”让它困惑了
    if final_docs:
        system_prompt = f"你是一个认知智能体。{tone_instruction}\n根据记忆片段回答。\n\n【记忆片段】:\n{{context}}"
        context_text = "\n\n".join([d.page_content for d in final_docs])
    else:
        system_prompt = f"你是一个专业的游戏运营认知智能体。{tone_instruction}\n请根据你的基础知识回答。"
        context_text = "（当前无外部记忆片段）"

    prompt = ChatPromptTemplate.from_template(system_prompt + "\n\n问题: {input}")
    res = (prompt | llm).invoke({"input": question, "context": context_text})
    answer = res.content if hasattr(res, "content") else str(res)

    return {
        "answer": answer,
        "context": final_docs,
        "generated_queries": search_queries,
        "used_alpha": final_alpha,
        "emotion": emotion_label,
        "entities": extracted_entities,
        "persona": player_persona,
    }


class LocalSearchInput(BaseModel):
    query: str = Field(description="需要搜索的具体问题字符串")


class LocalKnowledgeTool:
    def __init__(self, vectorstore, bm25_retriever, **kwargs):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.config = kwargs

    def _run_search(self, query: str) -> str:
        result = get_answer_complex(self.vectorstore, self.bm25_retriever, query, **self.config)
        docs_text = "\n".join([f"- {d.page_content}" for d in result["context"]])
        return f"【结论】: {result['answer']}\n【参考】:\n{docs_text}\n"

    def get_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self._run_search,
            name="local_knowledge_base",
            description="查询内部文档时调用。",
            args_schema=LocalSearchInput,
        )

