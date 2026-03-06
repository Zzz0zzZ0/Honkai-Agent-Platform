from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

print("🧠 [Perception] 正在激活基于 LLM 结构化输出的全新感知边缘系统...")

# 定义严格的数据输出骨架 (Schema)
class UserPerception(BaseModel):
    emotion: str = Field(description="用户情绪状态，必须是以下之一：positive, neutral, negative。如果是吐槽、抱怨，选择 negative。")
    entities: List[str] = Field(description="用户提到的核心实体，如角色名、游戏玩法等。")
    player_persona: List[str] = Field(description="根据用户发言推断的玩家画像标签，如 [强度党], [剧情党], [萌新], [零氪], [重氪] 等。")

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
        result = chain.invoke({"query": query})
        
        # 格式化兼容原有的实体提取返回值
        entities_formatted = [(e, "ENTITY") for e in result.entities]
        return result.emotion, entities_formatted, result.player_persona
        
    except Exception as e:
        print(f"感知模块提取异常: {e}")
        return "neutral", [], ["未知画像"]