from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import requests
from .utils import settings
import logging
from .vector_store import VectorStore
from django.core.files.storage import FileSystemStorage
from .utils.route_analyzer import RouteAnalyzer
import requests
logger = logging.getLogger(__name__)
# 在文件开头添加
route_analyzer = RouteAnalyzer()
# 初始化 VectorStore
vector_store = VectorStore()

def index(request):
    return render(request, 'chat/index.html')

@csrf_exempt
def chat(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            messages = data.get('messages', [])
            selected_model = data.get('model', 'llama3.2:latest')
            selected_mode = data.get('mode', 'path_planning')
            
            if not messages:
                return JsonResponse({"error": "No messages provided"}, status=400)
            
            last_message = messages[-1]['content']
            route_info = None
            
            # 如果是路径规划模式,先用模型分析起终点
            # 如果是路径规划模式,先用模型分析起终点
            if selected_mode == 'path_planning':
                # 修改路线分析提示词
                route_prompt = f"""
                你是一个专业的路线规划助手。请仔细分析用户的出行需求，并提供最合适的路线建议。

                用户输入: {last_message}

                请从以下几个方面分析用户需求：
                1. 基本信息提取：
                   - 起点位置
                   - 终点位置
                   - 途经点（如果有）
                   - 出发时间
                   - 期望到达时间（如果有）

                2. 用户偏好分析：
                   - 是否强调时间（包含"最快"、"尽快"、"赶时间"等词）
                   - 是否强调费用（包含"省钱"、"经济"、"便宜"等词）
                   - 是否强调距离（包含"最近"、"近路"等词）
                   - 是否关注路况（包含"堵车"、"路况"等词）

                3. 特殊要求识别：
                   - 是否需要避开高速
                   - 是否需要避开收费路段
                   - 是否需要避开拥堵路段

                请以JSON格式返回分析结果：
                {{
                    "route_info": {{
                        "start_point": "起点位置",
                        "end_point": "终点位置",
                        "waypoints": ["途经点1", "途经点2"],
                        "departure_time": "出发时间",
                        "arrival_time": "期望到达时间"
                    }},
                    "user_preferences": {{
                        "time_priority": true/false,      // 是否优先考虑时间
                        "cost_priority": true/false,      // 是否优先考虑费用
                        "distance_priority": true/false,  // 是否优先考虑距离
                        "traffic_priority": true/false    // 是否考虑实时路况
                    }},
                    "constraints": {{
                        "avoid_highways": true/false,    // 是否避开高速
                        "avoid_tolls": true/false,       // 是否避开收费
                        "avoid_congestion": true/false   // 是否避开拥堵
                    }},
                    "recommended_routes": [
                        {{
                            "type": "LEAST_TIME",         // 路线类型���LEAST_TIME/LEAST_FEE/LEAST_DISTANCE/REAL_TRAFFIC
                            "priority": 1,                // 优先级：1最高，数字越大优先级越低
                            "reason": "用户要求最快路线"   // 推荐原因
                        }},
                        {{
                            "type": "REAL_TRAFFIC",
                            "priority": 2,
                            "reason": "考虑实时路况避免拥堵"
                        }}
                    ],
                    "explanation": "根据您的需求，我建议优先考虑最快路线，同时为您提供一条考虑实时路况的备选路线..."
                }}

                分析说明：
                1. 必须根据用户的具体描述确定路线优先级
                2. 如果用户明确要求某种类型的路线，应将其设为最高优先级
                3. 如果用户提到多个需求（如"最快和最经济"），应同时推荐多条路线
                4. 每条推荐路线都要有明确的推荐理由
                5. 在 explanation 中详细解释推荐路线的原因和优势

                注意事项：
                1. 认真分析用户的每个关键词
                2. 如果用户说"推荐最经济和最快的路线"，必须在 recommended_routes 中包含这两种类型
                3. 推荐路线的顺序要符合用户的优先级要求
                4. 确保解释合理且符合用户需求

                示例：
                如果用户说"推荐最快和最经济的路线"，应该推荐：
                {{
                    "recommended_routes": [
                        {{
                            "type": "LEAST_TIME",
                            "priority": 1,
                            "reason": "满足用户对最快路线的需求"
                        }},
                        {{
                            "type": "LEAST_FEE",
                            "priority": 1,
                            "reason": "满足用户对最经济路线的需求"
                        }}
                    ]
                }}
                """
                
                # 调用本地模型分析
                analysis_request = {
                    "model": selected_model,
                    "prompt": route_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                    }
                }
                
                analysis_response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=analysis_request,
                    timeout=120,
                    headers={'Content-Type': 'application/json'}
                )
                
                if analysis_response.status_code == 200:
                    try:
                        analysis_result = json.loads(analysis_response.json()['response'])
                        start_point = analysis_result.get('start_point')
                        end_point = analysis_result.get('end_point')
                        
                        if start_point and end_point:
                            # 调用高德地图 API 获取地理编码
                            amap_key = settings.AMAP_API_KEY
                            geocode_url = "https://restapi.amap.com/v3/geocode/geo"
                            
                            # 获取起点坐标
                            start_response = requests.get(geocode_url, params={
                                "key": amap_key,
                                "address": start_point,
                                "city": "郑州"
                            })
                            start_data = start_response.json()
                            
                            # 获取终点坐标
                            end_response = requests.get(geocode_url, params={
                                "key": amap_key,
                                "address": end_point,
                                "city": "郑州"
                            })
                            end_data = end_response.json()
                            
                            route_data = None
                            if start_data['status'] == '1' and end_data['status'] == '1':
                                start_location = start_data['geocodes'][0]['location']
                                end_location = end_data['geocodes'][0]['location']
                                
                                # 获取路线规划
                                direction_url = "https://restapi.amap.com/v3/direction/walking"
                                route_response = requests.get(direction_url, params={
                                    "key": amap_key,
                                    "origin": start_location,
                                    "destination": end_location
                                })
                                route_data = route_response.json()
                        
                        route_info = {
                            "start_point": start_point,
                            "end_point": end_point,
                            "time": analysis_result.get('time'),
                            "requirements": analysis_result.get('requirements'),
                            "original_text": last_message,
                            "route_data": route_data if route_data and route_data['status'] == '1' else None
                        }
                    except json.JSONDecodeError:
                        logger.error("Failed to parse route analysis result")
                        route_info = None
            
            # 从知识库中检索相关信息
            knowledge_results = vector_store.query(last_message, n_results=3)
            knowledge_context = ""
            for result in knowledge_results['documents']:
                knowledge_context += f"Document: {result}\n"
            
            # 构建完整的上下文提示
            context = ""
            for msg in messages[:-1]:
                role = "User" if msg['role'] == 'user' else "Assistant"
                context += f"{role}: {msg['content']}\n"
            
            full_prompt = f"{knowledge_context}\n{context}User: {last_message}\nAssistant:"
            
            # 构建 Ollama API 请求
            ollama_request = {
                "model": selected_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "context_window": 4096,
                    "num_predict": 1000,
                }
            }
            
            # 调用 Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=ollama_request,
                timeout=120,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                response_data = response.json()
                ai_response = response_data.get('response', '').strip()
                
                # 修改引用信息的格式
                references = []
                for i, doc in enumerate(knowledge_results['documents'][0]):
                    references.append({
                        "index": i,
                        "document": f"文档 {i + 1}",
                        "content": doc
                    })
                
                formatted_response = {
                    "message": {
                        "role": "assistant",
                        "content": ai_response
                    },
                    "knowledge": references,
                    "route_info": route_info
                }
                return JsonResponse(formatted_response)
            else:
                error_msg = f"Ollama API error: {response.text}"
                return JsonResponse({"error": error_msg}, status=500)
                
        except Exception as e:
            logger.error(f"Error in chat view: {str(e)}")
            return JsonResponse({"error": str(e)}, status=500)
            
    return JsonResponse({"error": "Method not allowed"}, status=405)

def knowledge_page(request):
    """渲染知识库管理页面"""
    return render(request, 'chat/knowledge.html')

@csrf_exempt
@require_http_methods(["POST"])
def add_knowledge(request):
    """添加新知识到向量数据库"""
    try:
        data = json.loads(request.body)
        texts = data.get('texts', [])
        
        if not texts:
            return JsonResponse({"error": "No texts provided"}, status=400)
        
        vector_store.add_documents(texts)
        return JsonResponse({"message": "Knowledge added successfully"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_http_methods(["GET"])
def get_knowledge(request):
    """获取所有知识条目"""
    try:
        documents = vector_store.get_all_documents()
        return JsonResponse({"knowledge": documents})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def delete_knowledge(request):
    """删除指定的知识条目"""
    try:
        data = json.loads(request.body)
        index = data.get('index')
        
        if index is None:
            return JsonResponse({"error": "No index provided"}, status=400)
        
        success = vector_store.delete_document(index)
        if success:
            return JsonResponse({"message": "Knowledge deleted successfully"})
        else:
            return JsonResponse({"error": "Failed to delete knowledge"}, status=500)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def upload_file(request):
    """处理文件上传"""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)
        
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        
        # 保存文件
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)
        
        # 处理文件
        success = vector_store.add_file(file_path)
        
        # 删除临时文件
        fs.delete(filename)
        
        if success:
            return JsonResponse({"message": "File processed successfully"})
        else:
            return JsonResponse({"error": "Failed to process file"}, status=500)
            
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def route_plan(request):
    try:
        data = json.loads(request.body)
        start_point = data.get('start_point')
        end_point = data.get('end_point')
        
        if not start_point or not end_point:
            return JsonResponse({
                "error": "Missing start_point or end_point"
            }, status=400)
            
        # 调用高德地图 API 获取地理编码
        amap_key = settings.AMAP_API_KEY
        geocode_url = "https://restapi.amap.com/v3/geocode/geo"
        
        # 获取起点坐标
        start_response = requests.get(geocode_url, params={
            "key": amap_key,
            "address": start_point,
            "city": "郑州"
        })
        start_data = start_response.json()
        
        # 获取终点坐标
        end_response = requests.get(geocode_url, params={
            "key": amap_key,
            "address": end_point,
            "city": "郑州"
        })
        end_data = end_response.json()
        
        if start_data['status'] == '1' and end_data['status'] == '1':
            start_location = start_data['geocodes'][0]['location']
            end_location = end_data['geocodes'][0]['location']
            
            # 获取路线规划
            direction_url = "https://restapi.amap.com/v3/direction/driving"
            route_response = requests.get(direction_url, params={
                "key": amap_key,
                "origin": start_location,
                "destination": end_location,
                "extensions": "all",
                "strategy": 0  # 速度优先
            })
            route_data = route_response.json()
            
            if route_data['status'] == '1':
                return JsonResponse({
                    "success": True,
                    "data": route_data,
                    "start_location": start_location,
                    "end_location": end_location
                })
            else:
                return JsonResponse({
                    "error": "Failed to get route data",
                    "details": route_data
                }, status=500)
        else:
            return JsonResponse({
                "error": "Failed to geocode locations",
                "start_data": start_data,
                "end_data": end_data
            }, status=500)
            
    except json.JSONDecodeError:
        return JsonResponse({
            "error": "Invalid JSON data"
        }, status=400)
    except Exception as e:
        return JsonResponse({
            "error": str(e)
        }, status=500)