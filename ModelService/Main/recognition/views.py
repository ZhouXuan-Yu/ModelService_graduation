from django.http import JsonResponse
from .models import Recognition
import json
from transformers import AutoTokenizer, AutoModel
import torch
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

class ImageRecognitionView(APIView):
    def post(self, request):
        try:
            # 获取上传的图片和描述文本
            image_file = request.FILES.get('image')
            description = request.POST.get('description')

            # 1. 使用本地大模型提取关键词
            keywords = self.extract_keywords(description)
            
            # 2. 使用YOLO进行目标检测
            results = self.detect_objects(image_file, keywords)
            
            return JsonResponse({
                'status': 'success',
                'results': results
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })

    def extract_keywords(self, description):
        # 加载本地大模型
        tokenizer = AutoTokenizer.from_pretrained("your-local-model-path")
        model = AutoModel.from_pretrained("your-local-model-path")
        
        # 构建提示词
        prompt = f"""
        请从以下描述中提取关键特征词，格式为JSON：
        描述：{description}
        需要提取：
        1. 性别 (man/woman)
        2. 颜色
        3. 衣物类型
        """
        
        # 获取模型输出并解析
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        result = tokenizer.decode(outputs[0])
        
        # 解析JSON格式的关键词
        try:
            keywords = json.loads(result)
            return keywords
        except:
            return None

    def detect_objects(self, image_file, keywords):
        # 加载YOLO模型
        model = YOLO('yolov8n.pt')
        
        # 读取图片
        image = Image.open(image_file)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 进行目标检测
        results = model(image_np)
        
        # 根据关键词筛选结果
        filtered_results = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 获取类别和置信度
                cls = int(box.cls)
                conf = float(box.conf)
                
                # 如果检测到的是人，且符合关键词描述
                if model.names[cls] == 'person':
                    # 提取对应区域进行颜色分析
                    x1, y1, x2, y2 = box.xyxy[0]
                    roi = image_np[int(y1):int(y2), int(x1):int(x2)]
                    
                    # 分析该区域的主要颜色
                    color = self.analyze_color(roi)
                    
                    # 如果颜色匹配关键词中的颜色
                    if color.lower() == keywords['color'].lower():
                        filtered_results.append({
                            'box': box.xyxy[0].tolist(),
                            'confidence': conf,
                            'color': color
                        })
        
        return filtered_results

    def analyze_color(self, roi):
        # 简单的颜色分析示例
        # 这里可以根据需求实现更复杂的颜色分析算法
        mean_color = cv2.mean(roi)[:3]
        # 根据RGB值判断颜色
        # 这里需要完善颜色判断逻辑
        return "black" # 示例返回 