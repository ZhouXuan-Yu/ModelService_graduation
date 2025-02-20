from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from pathlib import Path
import os
from datetime import datetime
import logging
from typing import Dict, List, Optional
import torch
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
from app.utils.color_mapping import translate_color
from app.utils.image_analyzer import image_analyzer
from ultralytics import YOLO
import asyncio
import threading
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# 创建临时文件和输出目录
TEMP_DIR = Path("Main/temp/video/uploads")
OUTPUT_DIR = Path("Main/output/video")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 全局任务状态字典
processing_tasks = {}

class VideoProcessingTask:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.is_running = True
        self.progress = 0
        self.total_frames = 0
        self.processed_frames = 0
        self.output_path = None
        self.frame_results = []
        self.start_time = time.time()

def save_temp_file(file: UploadFile) -> Path:
    """保存上传的文件到临时目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = TEMP_DIR / f"{timestamp}_{file.filename}"
    
    with open(temp_file, "wb") as buffer:
        buffer.write(file.file.read())
    return temp_file

def cleanup_temp_files():
    """清理临时文件"""
    # 清理超过24小时的临时文件
    for file in TEMP_DIR.glob("*"):
        if file.is_file() and (datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)).days >= 1:
            file.unlink()
    
    # 清理超过24小时的输出文件
    for file in OUTPUT_DIR.glob("*"):
        if file.is_file() and (datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)).days >= 1:
            file.unlink()

class VideoAnalyzer:
    def __init__(self):
        self.image_analyzer = image_analyzer
        self.frame_skip = 2  # 每隔几帧处理一次
        
    async def process_video(self, input_path: str, output_path: str, task: VideoProcessingTask) -> Dict:
        """处理视频文件"""
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {input_path}")
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 更新任务信息
            task.total_frames = total_frames
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            # 分析结果
            frame_results = []
            frame_count = 0
            processed_count = 0
            
            while cap.isOpened() and task.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 每隔几帧处理一次
                if frame_count % self.frame_skip == 0:
                    try:
                        # 保存当前帧为临时图片
                        temp_frame_path = TEMP_DIR / f"frame_{frame_count}.jpg"
                        cv2.imwrite(str(temp_frame_path), frame)
                        
                        # 使用图像分析器处理帧
                        frame_result = await self.image_analyzer.analyze_image(str(temp_frame_path))
                        
                        # 检查是否检测到人物
                        if frame_result.get('num_persons', 0) > 0:
                            # 在帧上绘制分析结果
                            annotated_frame = self._draw_analysis(frame, frame_result)
                            
                            # 添加帧信息
                            frame_result['frame_number'] = frame_count
                            frame_result['timestamp'] = frame_count / fps
                            frame_results.append(frame_result)
                            
                            # 写入标注后的帧
                            out.write(annotated_frame)
                        else:
                            # 如果没有检测到人物，直接写入原始帧
                            out.write(frame)
                        
                        # 删除临时帧文件
                        temp_frame_path.unlink()
                        
                        processed_count += 1
                    except Exception as e:
                        logger.error(f"处理第 {frame_count} 帧时出错: {str(e)}")
                        out.write(frame)
                else:
                    out.write(frame)
                
                frame_count += 1
                
                # 更新进度
                task.progress = (frame_count / total_frames) * 100
                task.processed_frames = processed_count
                
                # 打印进度
                if frame_count % 30 == 0:
                    logger.info(f"处理进度: {task.progress:.1f}%")
            
            # 清理资源
            cap.release()
            out.release()
            
            # 更新任务状态
            task.output_path = output_path
            
            # 返回分析结果
            return {
                'total_frames': total_frames,
                'processed_frames': processed_count,
                'fps': fps,
                'duration': total_frames / fps,
                'resolution': {
                    'width': frame_width,
                    'height': frame_height
                },
                'frame_results': frame_results
            }
            
        except Exception as e:
            logger.error(f"处理视频时出错: {str(e)}")
            raise
    
    def _draw_analysis(self, frame: np.ndarray, analysis: Dict) -> np.ndarray:
        """在帧上绘制分析结果"""
        try:
            result_frame = frame.copy()
            
            # 绘制每个检测到的人物信息
            for person in analysis.get('persons', []):
                # 获取人脸边界框
                if person['face']['detected']:
                    x1, y1, x2, y2 = person['face']['bbox']
                    
                    # 绘制人脸框
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 添加文本信息
                    text_lines = []
                    
                    # 添加性别信息
                    if person['gender']['detected']:
                        gender_text = f"性别: {person['gender']['value']} ({person['gender']['confidence']:.2f})"
                        text_lines.append(gender_text)
                    
                    # 添加年龄信息
                    if person['age']['detected']:
                        age_text = f"年龄: {person['age']['value']:.1f} ({person['age']['confidence']:.2f})"
                        text_lines.append(age_text)
                    
                    # 添加衣着信息
                    if person['clothing']['upper']['detected']:
                        upper_text = f"上衣: {person['clothing']['upper']['color']} ({person['clothing']['upper']['confidence']:.2f})"
                        text_lines.append(upper_text)
                    
                    if person['clothing']['lower']['detected']:
                        lower_text = f"下装: {person['clothing']['lower']['color']} ({person['clothing']['lower']['confidence']:.2f})"
                        text_lines.append(lower_text)
                    
                    # 绘制文本
                    text_y = y1 - 10
                    for line in text_lines:
                        text_y -= 20
                        cv2.putText(result_frame, line, (x1, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return result_frame
            
        except Exception as e:
            logger.error(f"绘制分析结果时出错: {str(e)}")
            return frame

# 创建全局视频分析器实例
video_analyzer = VideoAnalyzer()

@router.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    description: str = Form(...)
):
    """分析上传的视频"""
    try:
        # 清理旧的临时文件
        cleanup_temp_files()
        
        # 验证文件类型
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="只接受视频文件")
        
        # 生成任务ID
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建任务对象
        task = VideoProcessingTask(task_id)
        processing_tasks[task_id] = task
        
        # 保存上传的视频
        temp_file = save_temp_file(video)
        
        try:
            # 生成输出文件路径
            output_filename = f"result_{task_id}_{video.filename}"
            output_file = OUTPUT_DIR / output_filename
            
            # 处理视频
            result = await video_analyzer.process_video(str(temp_file), str(output_file), task)
            
            if not result:
                raise HTTPException(status_code=500, detail="视频处理失败")
            
            # 构建结果视频的URL路径
            video_url = f"/output/video/{output_filename}"
            
            # 返回处理结果
            return JSONResponse(content={
                "message": "视频处理成功",
                "task_id": task_id,
                "video_path": video_url,
                "analysis_result": {
                    "total_frames": result['total_frames'],
                    "processed_frames": result['processed_frames'],
                    "fps": result['fps'],
                    "duration": result['duration'],
                    "resolution": result['resolution'],
                    "frame_results": result['frame_results']
                }
            })
            
        except Exception as e:
            logger.error(f"处理视频时出错: {str(e)}")
            raise HTTPException(status_code=500, detail=f"处理视频时出错: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 清理临时文件
        if 'temp_file' in locals():
            temp_file.unlink(missing_ok=True)

@router.post("/stop/{task_id}")
async def stop_processing(task_id: str):
    """停止视频处理任务"""
    if task_id in processing_tasks:
        task = processing_tasks[task_id]
        task.is_running = False
        return {"message": "正在停止处理任务"}
    raise HTTPException(status_code=404, detail="任务不存在")

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id in processing_tasks:
        task = processing_tasks[task_id]
        return {
            "task_id": task_id,
            "is_running": task.is_running,
            "progress": task.progress,
            "total_frames": task.total_frames,
            "processed_frames": task.processed_frames,
            "elapsed_time": time.time() - task.start_time,
            "output_path": task.output_path
        }
    raise HTTPException(status_code=404, detail="任务不存在")

@router.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

@router.get("/track")
async def get_tracking():
    return []

@router.post("/track")
async def start_tracking(video: UploadFile):
    """开始视频追踪"""
    
@router.get("/status/{tracking_id}")
async def get_tracking_status(tracking_id: str):
    """获取追踪状态"""
    
@router.get("/history")
async def get_tracking_history():
    """获取追踪历史""" 