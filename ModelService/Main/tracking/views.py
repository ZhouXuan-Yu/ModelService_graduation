from django.http import JsonResponse
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO
import json
import cv2
import asyncio
import numpy as np
from PIL import Image
import torch

class VideoTrackingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.tracking_id = self.scope['url_route']['kwargs']['tracking_id']
        await self.accept()
        
    async def disconnect(self, close_code):
        # 清理资源
        pass

    async def receive(self, text_data):
        # 处理客户端消息
        pass

class VideoTrackingView(APIView):
    def post(self, request):
        try:
            video_file = request.FILES.get('video')
            description = request.POST.get('description')

            # 生成追踪ID
            tracking_id = str(uuid.uuid4())

            # 保存视频文件
            video_path = f'media/videos/{tracking_id}.mp4'
            with open(video_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)

            # 启动异步追踪任务
            asyncio.create_task(self.process_video(tracking_id, video_path, description))

            return JsonResponse({
                'status': 'success',
                'trackingId': tracking_id
            })

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })

    async def process_video(self, tracking_id, video_path, description):
        # 加载YOLO模型
        model = YOLO('yolov8n.pt')
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 初始化追踪器
        tracker = model.track(source=video_path, persist=True, verbose=False)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 进行目标检测和追踪
            results = tracker(frame)
            
            # 获取追踪框
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                
                # 发送追踪结果到WebSocket
                tracking_data = {
                    'type': 'tracking_result',
                    'boxes': boxes.tolist(),
                    'track_ids': track_ids.tolist(),
                    'progress': int((frame_count / total_frames) * 100)
                }
                
                channel_layer = get_channel_layer()
                await channel_layer.group_send(
                    f'tracking_{tracking_id}',
                    {
                        'type': 'send_tracking_update',
                        'data': json.dumps(tracking_data)
                    }
                )
            
            frame_count += 1
            
        # 发送完成消息
        await channel_layer.group_send(
            f'tracking_{tracking_id}',
            {
                'type': 'send_tracking_update',
                'data': json.dumps({
                    'type': 'tracking_complete'
                })
            }
        )
        
        cap.release() 