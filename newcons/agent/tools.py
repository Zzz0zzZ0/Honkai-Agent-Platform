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
    log_file = "community_feedback_log.csv"
    if not os.path.exists(log_file):
        return "【舆情大盘】当前数据库为空。"
    try:
        df = pd.read_csv(log_file)
        neg_df = df[df["Emotion"] == "negative"]
        recent_complaints = "\n".join(
            [
                f"- {row['Player_Query']} (画像: {row.get('Player_Persona', '未知')})"
                for _, row in neg_df.tail(3).iterrows()
            ]
        )
        return (
            f"【舆情大盘分析】目前共收录 {len(df)} 条反馈，高危负面 {len(neg_df)} 条。\n"
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

