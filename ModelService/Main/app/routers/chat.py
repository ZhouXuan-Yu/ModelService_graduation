from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
import requests

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "gemma2:2b"
    type: str = "general"

@router.post("/completions")
async def chat_completion(request: ChatRequest):
    """聊天完成"""
    try:
        # 调用 Ollama API
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": request.model,
                "prompt": request.messages[-1].content,
                "stream": False
            }
        )
        
        if ollama_response.status_code == 200:
            return {
                "message": ollama_response.json()["response"]
            }
        else:
            raise HTTPException(status_code=500, detail="Ollama API 调用失败")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/completions/history")
async def get_chat_history(type: str = "general"):
    """获取聊天历史"""
    try:
        # 这里可以添加数据库查询逻辑
        history = []  # 暂时返回空列表
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))