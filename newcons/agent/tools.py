import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool


# ==========================================
# 游戏业务专属 Agent 工具（纯工具函数）
# ==========================================


@tool
def star_rail_gacha_calculator(current_pity: int, target_copies: int) -> str:
    """当用户询问抽卡规划、大保底计算时调用此工具"""
    cost_per_pull = 160
    expected_pulls_per_copy = 94
    total_expected_pulls = max(0, (target_copies * expected_pulls_per_copy) - current_pity)
    total_jade_needed = total_expected_pulls * cost_per_pull
    return (
        f"【运营数值推演】目标抽取 {target_copies} 只，当前水位 {current_pity} 抽。"
        f"预计还需 {total_expected_pulls} 抽，折合星琼约 {total_jade_needed}。"
    )


@tool
def analyze_community_feedback() -> str:
    """当运营人员要求总结今天的社区舆情、玩家负面反馈时调用此工具"""
    import sqlite3
    db_file = "community_feedback_log.db"
    if not os.path.exists(db_file):
        return "【舆情大盘】当前数据库为空。"
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM feedback_logs")
        total_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM feedback_logs WHERE Emotion = 'negative'")
        neg_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT Player_Query, Player_Persona FROM feedback_logs WHERE Emotion = 'negative' ORDER BY id DESC LIMIT 3")
        recent_rows = cursor.fetchall()
        
        recent_complaints = "\n".join(
            [
                f"- {row[0]} (画像: {row[1]})"
                for row in recent_rows
            ]
        )
        conn.close()
        return (
            f"【舆情大盘分析】目前共收录 {total_count} 条反馈，高危负面 {neg_count} 条。\n"
            f"近期核心槽点：\n{recent_complaints}"
        )
    except Exception as e:
        return f"读取舆情数据库失败: {str(e)}"


@tool
def generate_pr_announcement(issue_summary: str, compensation: str = "300星琼") -> str:
    """
    当运营人员要求撰写滑轨公告、道歉信、或者针对玩家的大规模吐槽要求出具公关文案时调用此工具。
    输入参数：
    - issue_summary (str): 玩家近期吐槽的核心问题总结。
    - compensation (str): 拟定的补偿方案，默认 300星琼。
    """
    prompt = (
        "你现在是《崩坏：星穹铁道》列车长帕姆。"
        f"针对玩家问题：【{issue_summary}】写一份真诚的滑轨道歉公告，并宣布全服补偿：{compensation}。"
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    res = llm.invoke(prompt)
    return f"【已自动生成公关滑轨草案，请审核】\n\n{res.content}"

