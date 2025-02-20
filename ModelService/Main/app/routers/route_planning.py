from fastapi import APIRouter, HTTPException, Request
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
import requests
import json
from ..utils.vector_store import VectorStore
import re
from datetime import datetime
import os

# 从环境变量获取API密钥
AMAP_KEY = os.getenv('AMAP_KEY', '5c98219ee72ff8b122e46b8167333eb9')

router = APIRouter()
vector_store = VectorStore()

class RoutePreferences(BaseModel):
    """路线偏好设置"""
    avoid_highways: bool = Field(default=False, description="避开高速")
    avoid_tolls: bool = Field(default=False, description="避开收费")
    avoid_congestion: bool = Field(default=True, description="避开拥堵")

class RouteInfo(BaseModel):
    """路线信息"""
    start_point: str = Field(..., description="起点")
    end_point: str = Field(..., description="终点")
    waypoints: List[str] = Field(default=[], description="途经点")
    departure_time: str = Field(default="", description="出发时间")
    arrival_time: str = Field(default="", description="到达时间")
    route_type: str = Field(default="LEAST_TIME", description="路线类型")

class RecommendedRoute(BaseModel):
    """推荐路线"""
    type: str = Field(..., description="路线类型")
    name: str = Field(..., description="路线名称")
    reason: str = Field(..., description="推荐原因")

class RouteRequest(BaseModel):
    """路线规划请求"""
    text: str = Field(..., description="用户输入文本")
    model: str = Field(default="gemma2:2b", description="使用的模型")
    historical_route: Optional[Dict] = Field(default=None, description="历史路线")

class RouteResponse(BaseModel):
    """路线规划响应"""
    success: bool = Field(..., description="是否成功")
    route_data: Optional[Dict] = Field(default=None, description="路线数据")
    error: Optional[str] = Field(default=None, description="错误信息")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="时间戳")

ROUTE_PROMPT = """你是一个专业的路线规划助手。请分析用户的出行需求，并返回JSON格式的路线规划。

用户输入: {text}

历史路线: {historical_context}

请按照以下格式返回JSON：
{{
    "recommended_routes": [
        {{
            "type": "LEAST_TIME",
            "name": "推荐路线",
            "reason": "最快到达目的地"
        }},
        {{
            "type": "LEAST_FEE",
            "name": "备选路线",
            "reason": "费用较低"
        }}
    ],
    "route_info": {{
        "start_point": "起点",
        "end_point": "终点",
        "waypoints": [],
        "departure_time": "",
        "arrival_time": "",
        "route_type": "LEAST_TIME"
    }},
    "preferences": {{
        "avoid_highways": false,
        "avoid_tolls": false,
        "avoid_congestion": true
    }}
}}
"""

@router.post("/plan", response_model=RouteResponse)
async def create_route_plan(request: RouteRequest) -> RouteResponse:
    """创建路线规划"""
    try:
        print(f"收到路线规划请求: {request}")
        
        # 获取相关的历史路线
        similar_routes = vector_store.query(request.text, n_results=1)
        historical_context = ""
        
        if similar_routes and similar_routes['documents']:
            historical_context = similar_routes['documents'][0]
        
        # 调用 Ollama 分析用户需求
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": request.model,
                "prompt": ROUTE_PROMPT.format(
                    text=request.text,
                    historical_context=historical_context
                ),
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.1
                }
            },
            timeout=400  # 设置超时时间为400秒
        )
        
        if ollama_response.status_code != 200:
            return RouteResponse(
                success=False,
                error=f"AI模型调用失败: {ollama_response.text}"
            )
        
        try:
            response_json = ollama_response.json()
            response_text = response_json["response"].strip()
            
            # 提取JSON内容
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx]
                response_text = response_text.replace('\n', ' ').replace('\r', ' ')
                response_text = re.sub(r'\s+', ' ', response_text)
                response_text = re.sub(r',\s*}', '}', response_text)
                response_text = re.sub(r',\s*]', ']', response_text)
                
                try:
                    route_analysis = json.loads(response_text)
                except json.JSONDecodeError:
                    response_text = response_text.replace("'", '"')
                    response_text = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', response_text)
                    route_analysis = json.loads(response_text)
                
                # 验证和补充必要字段
                if 'route_info' not in route_analysis:
                    route_analysis['route_info'] = {}
                
                route_info = route_analysis['route_info']
                
                # 验证必要字段
                if not route_info.get('start_point') or not route_info.get('end_point'):
                    return RouteResponse(
                        success=False,
                        error="缺少起点或终点信息"
                    )
                
                # 补充默认值
                route_info.setdefault('waypoints', [])
                route_info.setdefault('departure_time', '')
                route_info.setdefault('arrival_time', '')
                route_info.setdefault('route_type', 'LEAST_TIME')
                
                # 验证路线类型
                valid_types = ['LEAST_TIME', 'LEAST_FEE', 'LEAST_DISTANCE']
                if route_info['route_type'] not in valid_types:
                    route_info['route_type'] = 'LEAST_TIME'
                
                # 确保推荐路线存在
                if 'recommended_routes' not in route_analysis:
                    route_analysis['recommended_routes'] = [
                        {
                            "type": route_info['route_type'],
                            "name": "推荐路线",
                            "reason": "根据您的需求推荐"
                        }
                    ]
                
                # 确保偏好设置存在
                if 'preferences' not in route_analysis:
                    route_analysis['preferences'] = {
                        "avoid_highways": False,
                        "avoid_tolls": False,
                        "avoid_congestion": True
                    }
                
                return RouteResponse(
                    success=True,
                    route_data=route_analysis
                )
            else:
                return RouteResponse(
                    success=False,
                    error="无法找到有效的JSON数据"
                )
            
        except json.JSONDecodeError as e:
            return RouteResponse(
                success=False,
                error=f"JSON解析错误: {str(e)}"
            )
            
    except requests.RequestException as e:
        return RouteResponse(
            success=False,
            error=f"请求错误: {str(e)}"
        )
    except Exception as e:
        return RouteResponse(
            success=False,
            error=f"系统错误: {str(e)}"
        )

@router.get("/history")
async def get_route_history():
    """获取路线历史"""
    try:
        documents = vector_store.get_all_documents()
        return {
            "success": True,
            "history": documents,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history/{index}")
async def delete_route_history(index: int):
    """删除历史路线"""
    try:
        success = vector_store.delete_document(index)
        return {
            "success": success,
            "message": "删除成功" if success else "删除失败",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_location(address: str) -> str:
    """获取地理编码"""
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/geocode/geo",
            params={
                "key": AMAP_KEY,
                "address": address
            },
            timeout=400  # 设置超时时间为400秒
        )
        data = response.json()
        if data["status"] == "1" and data["geocodes"]:
            return data["geocodes"][0]["location"]
        return ""
    except Exception as e:
        logger.error(f"地理编码请求失败: {str(e)}")
        return ""

async def get_route_plan(
    start: str,
    end: str,
    preferences: Dict,
    route_type: str
) -> Dict:
    """获取路线规划详情"""
    try:
        strategy = {
            "LEAST_TIME": 0,
            "LEAST_FEE": 1,
            "LEAST_DISTANCE": 2
        }.get(route_type, 0)
        
        response = requests.get(
            "https://restapi.amap.com/v3/direction/driving",
            params={
                "key": AMAP_KEY,  # 使用环境变量中的API密钥
                "origin": start,
                "destination": end,
                "strategy": strategy,
                "extensions": "all"
            },
            timeout=400  # 设置超时时间为400秒
        )
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"路线规划请求失败: {str(e)}")

# 添加OPTIONS方法支持
@router.options("/plan")
async def options_route_plan():
    return {"message": "OK"}