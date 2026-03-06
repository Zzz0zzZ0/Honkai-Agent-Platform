from typing import Annotated, TypedDict
from langchain_core.messages import SystemMessage, AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools.tavily_search import TavilySearchResults
from engine.rag_pipeline import LocalKnowledgeTool
from agent.tools import (
    star_rail_gacha_calculator,
    analyze_community_feedback,
    generate_pr_announcement,
)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def build_graph_agent(vectorstore=None, bm25_retriever=None, **kwargs):
    web_search_tool = TavilySearchResults(max_results=3, description="查询外部实时信息、游戏攻略时使用。")
    
    # 装载四大神器
    tools = [
        web_search_tool,
        star_rail_gacha_calculator,
        analyze_community_feedback,
        generate_pr_announcement
    ]
    
    if vectorstore is not None and bm25_retriever is not None:
        local_tool_instance = LocalKnowledgeTool(vectorstore, bm25_retriever, **kwargs)
        tools.append(local_tool_instance.get_tool())

    model_type = kwargs.get("model_type", "cloud")
    temp_val = kwargs.get("temp_param", 0.1)

    if model_type == "local":
        llm = ChatOllama(model="qwen3:8b", temperature=temp_val)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp_val, safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE})
    
    llm_with_tools = llm.bind_tools(tools)

    sys_msg = SystemMessage(content=(
        "你是一个高级游戏运营Agent。请遵循原则：\n"
        "1. 玩家抽卡计算，调用 star_rail_gacha_calculator。\n"
        "2. 总结玩家吐槽与负面舆情，调用 analyze_community_feedback。\n"
        "3. 要求写致歉公告、滑轨文案，调用 generate_pr_announcement。\n"
        "4. 查询外网攻略/新闻，使用 tavily_search_results_json。\n"
    ))

    def call_model(state: AgentState):
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]} 

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()