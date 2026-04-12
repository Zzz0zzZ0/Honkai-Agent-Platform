from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import json
import re

print("🧠 [Perception] 正在激活基于 LLM 结构化输出的全新感知边缘系统...")

# 定义严格的数据输出骨架 (Schema)
class UserPerception(BaseModel):
    emotion: str = Field(description="用户情绪状态，必须是以下之一：positive, neutral, negative。如果是吐槽、抱怨，选择 negative。")
    entities: List[str] = Field(description="用户提到的核心实体，如角色名、游戏玩法等。")
    player_persona: List[str] = Field(description="根据用户发言推断的玩家画像标签，如 [强度党], [剧情党], [萌新], [零氪], [重氪] 等。")
    game_entities: List[str] = Field(description="提取玩家发言中提到的游戏专有名词、角色名、物品名等实体（如：'黄泉', '混沌回忆', '星琼'）")
def analyze_user_query(query: str, model_type="cloud", temp_val=0.1):
    """统一感知接口：一次性提取情感、实体与玩家画像"""
    try:
        if model_type == "local":
            llm = ChatOllama(model="qwen3:8b", temperature=temp_val)
        else:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp_val)
        
        # 强制 LLM 按照 Pydantic 类输出结构化数据
        structured_llm = llm.with_structured_output(UserPerception)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的游戏玩家行为分析专家。请仔细体会玩家发言中的潜台词，提取情绪、关键实体，并打上精准的玩家画像标签。"),
            ("human", "{query}")
        ])
        
        chain = prompt | structured_llm
        
        try:
            result = chain.invoke({"query": query})
            
            # 格式化兼容原有的实体提取返回值，并合并 entities 和 game_entities
            all_entities = list(set(result.entities + result.game_entities))
            entities_formatted = [(e, "ENTITY") for e in all_entities]
            return result.emotion, entities_formatted, result.player_persona
        
        except Exception as structure_e:
            print(f"结构化输出解析失败，尝试正则表达式 fallback: {structure_e}")
            
            # Fallback 机制：通过普通 prompt 让 LLM 返回 JSON 字符串并硬解
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个专业的游戏玩家行为分析专家。请仔细体会玩家发言中的潜台词，提取情绪、关键实体，并打上精准的玩家画像标签。\n"
                           "请必须返回严格的JSON格式字符串，包含以下字段：\n"
                           "- emotion (字符串: 'positive', 'neutral', 'negative')\n"
                           "- entities (字符串数组: 核心实体)\n"
                           "- player_persona (字符串数组: 玩家画像)\n"
                           "- game_entities (字符串数组: 游戏专有名词)"),
                ("human", "{query}")
            ])
            fallback_chain = fallback_prompt | llm
            raw_res = fallback_chain.invoke({"query": query}).content
            
            json_match = re.search(r'\{.*\}', raw_res, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
                emotion = parsed_json.get("emotion", "neutral")
                entities = parsed_json.get("entities", [])
                game_entities = parsed_json.get("game_entities", [])
                player_persona = parsed_json.get("player_persona", ["未知画像"])
                
                all_entities = list(set(entities + game_entities))
                entities_formatted = [(e, "ENTITY") for e in all_entities]
                return emotion, entities_formatted, player_persona
            else:
                raise ValueError("正则表达式也无法提取出合法的 JSON")
        
    except Exception as e:
        print(f"感知模块提取异常: {e}")
        return "neutral", [], ["未知画像"]