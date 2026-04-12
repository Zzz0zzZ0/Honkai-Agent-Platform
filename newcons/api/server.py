from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import tempfile
import os
import pandas as pd
import json  # 使用json替代ast
import sys
import threading  # 添加线程锁
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.vector_store import build_hybrid_knowledge_base, visualize_semantic_space
from engine.rag_pipeline import get_answer_complex, log_negative_feedback_sync
from agent.graph_brain import build_graph_agent
from langchain_community.retrievers import BM25Retriever


app = FastAPI(title="Cognitive Agent API", description="游戏智能运营大盘后台引擎")

# 添加线程锁保护全局状态
global_memory = {
    "vectorstore": None,
    "bm25": None,
    "all_splits": [],
}
memory_lock = threading.Lock()

# 文件上传配置
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.pdf', '.txt'}


class ChatRequest(BaseModel):
    query: str
    use_agent: bool = True
    model_type: str = "cloud"
    use_auto_alpha: bool = True
    alpha: float = 0.5
    use_emotion: bool = True
    # [新增] 恢复调节参数
    k_param: int = 3
    temp_param: float = 0.1

class FeedbackRequest(BaseModel):
    arm_idx: int
    context_vec: list
    reward: float


@app.post("/upload_memory")
async def upload_memory(file: UploadFile = File(...)):
    """处理前端上传的文档，注入全局记忆并生成可视化数据"""
    try:
        # 验证文件扩展名
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型。仅支持: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # 读取文件内容并验证大小
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件过大。最大允许 {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # 基本内容验证（检查是否为空或全是空白字符）
        if not file_content or not file_content.strip():
            raise HTTPException(status_code=400, detail="文件内容为空")
        
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_ext
        ) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name

        vs, new_splits, count = build_hybrid_knowledge_base(tmp_path)
        
        # 使用线程锁保护全局状态更新
        with memory_lock:
            global_memory["vectorstore"] = vs
            global_memory["all_splits"].extend(new_splits)
            
            if global_memory["all_splits"]:
                global_bm25 = BM25Retriever.from_documents(global_memory["all_splits"])
                global_bm25.k = 10
                global_memory["bm25"] = global_bm25

        # 生成可视化 2D 坐标数据，并转换成 JSON 格式发给前端
        viz_df = visualize_semantic_space(vs)
        viz_data_json = viz_df.to_dict(orient="records") if viz_df is not None else []

        os.remove(tmp_path)
        return {
            "status": "success",
            "message": f"记忆固化成功，共 {count} 个片段。",
            "viz_data": viz_data_json,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    """处理所有的对话与 Agent 路由请求"""

    def clean_llm_output(raw_output):
        # 使用json.loads替代ast.literal_eval，更安全
        if isinstance(raw_output, str) and raw_output.strip().startswith("[{"):
            try:
                raw_output = json.loads(raw_output.strip())
            except json.JSONDecodeError:
                # 如果JSON解析失败，返回原始字符串
                pass

        # 应对列表结构提取纯文本
        if isinstance(raw_output, list):
            texts = [
                item.get("text", "")
                for item in raw_output
                if isinstance(item, dict) and "text" in item
            ]
            if texts:
                return "".join(texts)

        return str(raw_output)

    try:
        if req.use_agent:
            # 使用线程锁保护全局状态读取
            with memory_lock:
                vectorstore = global_memory["vectorstore"]
                bm25 = global_memory["bm25"]
            
            # 1. 启动 Agent 执行复杂工具流与多步推理
            agent_app = build_graph_agent(
                vectorstore,
                bm25,
                model_type=req.model_type,
                use_emotion=req.use_emotion,
                use_auto_alpha=req.use_auto_alpha,
                temp_param=req.temp_param,
                k_param=req.k_param,
            )
            response = await run_in_threadpool(agent_app.invoke, {"messages": [("user", req.query)]})

            # 清洗 Agent 输出的乱码
            raw_content = response["messages"][-1].content
            answer = clean_llm_output(raw_content)

            # 提取思考流
            thoughts = []
            for msg in response["messages"][1:-1]:
                if msg.type == "ai" and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        thoughts.append(f"🛠️ 调用工具: {tc['name']}")

            # 2. 旁路提取玩家画像标签与 LinUCB 特征
            persona_tags = []
            arm_idx = -1
            context_vec = []
            if req.use_emotion:
                try:
                    temp_result = await run_in_threadpool(
                        get_answer_complex,
                        vectorstore,
                        bm25,
                        req.query,
                        model_type=req.model_type,
                        alpha=req.alpha,
                        use_auto_alpha=req.use_auto_alpha,
                        use_emotion=True,
                        k_param=req.k_param,
                        temp_param=req.temp_param,
                    )
                    persona_tags = temp_result.get("persona", [])
                    arm_idx = temp_result.get("arm_idx", -1)
                    context_vec = temp_result.get("context_vec", [])
                    if temp_result.get("emotion") == "negative":
                       background_tasks.add_task(log_negative_feedback_sync, req.query, "negative", "|".join(persona_tags))
                except Exception as e:
                    print(f"旁路画像与权重特征提取失败: {e}")

            return {
                "answer": answer,
                "thoughts": thoughts,
                "persona": persona_tags,
                "arm_idx": arm_idx,
                "context_vec": context_vec,
            }

        else:
            # 使用线程锁保护全局状态读取
            with memory_lock:
                vectorstore = global_memory["vectorstore"]
                bm25 = global_memory["bm25"]
                
            result = await run_in_threadpool(
                get_answer_complex,
                vectorstore,
                bm25,
                req.query,
                model_type=req.model_type,
                alpha=req.alpha,
                use_auto_alpha=req.use_auto_alpha,
                use_emotion=req.use_emotion,
                k_param=req.k_param,
                temp_param=req.temp_param,
            )

            if result.get("emotion") == "negative":
                 persona_tags = result.get("persona", [])
                 background_tasks.add_task(log_negative_feedback_sync, req.query, "negative", "|".join(persona_tags))

            raw_ans = result["answer"]
            clean_ans = clean_llm_output(raw_ans)

            return {
                "answer": clean_ans,
                "thoughts": [],
                "persona": result.get("persona", []),
                "arm_idx": result.get("arm_idx", -1),
                "context_vec": result.get("context_vec", []),
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def explicit_feedback(req: FeedbackRequest):
    """接收前端传回的点赞或点踩，作为 LinUCB 的显式奖励"""
    try:
        from algorithms.linucb import linucb_agent
        linucb_agent.update(req.arm_idx, req.context_vec, req.reward)
        return {"status": "success", "message": "Feedback received and LinUCB updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

