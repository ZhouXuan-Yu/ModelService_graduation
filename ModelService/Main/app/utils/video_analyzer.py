import cv2
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import logging
import mediapipe as mp
from app.utils.color_mapping import translate_color
from ultralytics import YOLO  # 修改导入语句
from app.utils.image_analyzer import image_analyzer
from datetime import datetime
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self):
        self.models = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        self.image_analyzer = image_analyzer
        self.frame_skip = 2  # 每隔几帧处理一次
        self.load_models()
        
    def load_models(self):
        """加载所有需要的模型"""
        try:
            # 获取项目根目录
            ROOT_DIR = Path(__file__).parent.parent.parent
            MODEL_DIR = ROOT_DIR / "model/output"
            
            # 使用 YOLOv8 加载人脸检测模型
            face_model_path = str(MODEL_DIR / 'face_detection/train2/weights/best.pt')
            self.models['face'] = YOLO(face_model_path)
            
            # 使用 YOLOv8 加载性别识别模型
            gender_model_path = str(MODEL_DIR / 'gender_classification/train/weights/best.pt')
            self.models['gender'] = YOLO(gender_model_path)
            
            try:
                # 加载年龄预测模型
                age_model_path = str(MODEL_DIR / 'age_estimation/weights/best.pt')
                logger.info(f"正在加载年龄预测模型: {age_model_path}")
                age_checkpoint = torch.load(age_model_path, map_location=self.device)
                
                # 打印模型结构以便调试
                logger.info(f"年龄预测模型结构: {type(age_checkpoint)}")
                if isinstance(age_checkpoint, dict):
                    logger.info(f"年龄预测模型键值: {age_checkpoint.keys()}")
                
                # 处理年龄预测模型
                if isinstance(age_checkpoint, dict):
                    # 尝试所有可能的键名
                    possible_keys = ['model', 'state_dict', 'model_state_dict', 'net', 'net_state_dict']
                    model_loaded = False
                    for key in possible_keys:
                        if key in age_checkpoint:
                            self.models['age'] = age_checkpoint[key]
                            logger.info(f"使用键 '{key}' 加载年龄预测模型")
                            model_loaded = True
                            break
                    
                    if not model_loaded:
                        logger.warning("未找到模型状态字典，使用整个检查点")
                        self.models['age'] = age_checkpoint
                else:
                    self.models['age'] = age_checkpoint
                
                # 加载颜色识别模型
                color_model_path = str(MODEL_DIR / 'color_classification/best_model.pth')
                logger.info(f"正在加载颜色识别模型: {color_model_path}")
                color_checkpoint = torch.load(color_model_path, map_location=self.device)
                
                # 打印模型结构以便调试
                logger.info(f"颜色识别模型结构: {type(color_checkpoint)}")
                if isinstance(color_checkpoint, dict):
                    logger.info(f"颜色识别模型键值: {color_checkpoint.keys()}")
                
                # 处理颜色识别模型
                if isinstance(color_checkpoint, dict):
                    # 尝试所有可能的键名
                    possible_keys = ['model', 'state_dict', 'model_state_dict', 'net', 'net_state_dict']
                    model_loaded = False
                    for key in possible_keys:
                        if key in color_checkpoint:
                            self.models['color'] = color_checkpoint[key]
                            logger.info(f"使用键 '{key}' 加载颜色识别模型")
                            model_loaded = True
                            break
                    
                    if not model_loaded:
                        logger.warning("未找到模型状态字典，使用整个检查点")
                        self.models['color'] = color_checkpoint
                else:
                    self.models['color'] = color_checkpoint
                
                # 将模型设置为评估模式
                for model_name in ['age', 'color']:
                    if self.models[model_name] is not None:
                        if hasattr(self.models[model_name], 'eval'):
                            self.models[model_name].eval()
                            logger.info(f"{model_name} 模型已设置为评估模式")
                        else:
                            logger.warning(f"{model_name} 模型没有eval方法")
                    
            except Exception as e:
                logger.error(f"加载PyTorch模型时出错: {str(e)}")
                logger.exception("详细错误信息:")
                # 如果加载失败，设置为None并继续
                self.models['age'] = None
                self.models['color'] = None
            
            logger.info("所有模型加载完成")
            
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            logger.exception("详细错误信息:")
            raise
            
    def preprocess_image(self, img):
        """预处理图像"""
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)
        return img_tensor.unsqueeze(0).to(self.device)
        
    def detect_faces(self, frame):
        """检测人脸"""
        try:
            results = self.models['face'](frame)
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                return boxes.data.cpu().numpy()  # 返回边界框数据
            return []
        except Exception as e:
            logger.error(f"人脸检测失败: {str(e)}")
            return []
            
    def predict_gender(self, face_img):
        """预测性别"""
        try:
            # 确保输入是RGB格式的PIL图像
            if isinstance(face_img, np.ndarray):
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = Image.fromarray(face_img)
            elif not isinstance(face_img, Image.Image):
                raise ValueError("输入必须是PIL图像或numpy数组")
            
            # 转换为numpy数组
            face_array = np.array(face_img)
            
            # 使用YOLO模型预测
            results = self.models['gender'](face_array)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                gender = "男" if cls == 0 else "女"
                return gender, conf
            
            return "未知", 0.0
            
        except Exception as e:
            logger.error(f"性别预测失败: {str(e)}")
            return "未知", 0.0
            
    def predict_age(self, face_img):
        """预测年龄"""
        try:
            face_tensor = self.preprocess_image(face_img)
            with torch.no_grad():
                output = self.models['age'](face_tensor)
                age = float(output.squeeze().cpu().numpy())
                return max(0, min(100, age))  # 限制在0-100岁范围内
        except Exception as e:
            logger.error(f"年龄预测失败: {str(e)}")
            return 0
            
    def analyze_clothing(self, person_img):
        """分析衣服颜色"""
        try:
            # 确保输入是RGB格式
            if isinstance(person_img, np.ndarray):
                person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            
            # 使用MediaPipe分割人体区域
            results = self.pose.process(person_img)
            if not results.pose_landmarks:
                return {"upper": ("未知", 0.0), "lower": ("未知", 0.0)}
            
            h, w = person_img.shape[:2]
            landmarks = results.pose_landmarks.landmark
            
            # 估计上衣区域
            upper_y1 = max(0, int(landmarks[11].y * h))  # 肩膀
            upper_y2 = min(h, int(landmarks[23].y * h))  # 臀部
            upper_x1 = max(0, int(min(landmarks[11].x, landmarks[12].x) * w) - 20)
            upper_x2 = min(w, int(max(landmarks[11].x, landmarks[12].x) * w) + 20)
            
            # 估计下装区域
            lower_y1 = max(0, int(landmarks[23].y * h))  # 臀部
            lower_y2 = min(h, int(landmarks[27].y * h))  # 膝盖
            lower_x1 = max(0, int(min(landmarks[23].x, landmarks[24].x) * w) - 20)
            lower_x2 = min(w, int(max(landmarks[23].x, landmarks[24].x) * w) + 20)
            
            # 提取区域
            if upper_y2 > upper_y1 and upper_x2 > upper_x1:
                upper_region = person_img[upper_y1:upper_y2, upper_x1:upper_x2]
            else:
                upper_region = np.array([])
                
            if lower_y2 > lower_y1 and lower_x2 > lower_x1:
                lower_region = person_img[lower_y1:lower_y2, lower_x1:lower_x2]
            else:
                lower_region = np.array([])
            
            # 分析颜色
            upper_color = self.predict_color(upper_region) if upper_region.size > 0 else ("未知", 0.0)
            lower_color = self.predict_color(lower_region) if lower_region.size > 0 else ("未知", 0.0)
            
            return {
                "upper": upper_color,
                "lower": lower_color
            }
            
        except Exception as e:
            logger.error(f"服装分析失败: {str(e)}")
            return {"upper": ("未知", 0.0), "lower": ("未知", 0.0)}
            
    def predict_color(self, img):
        """预测颜色"""
        try:
            if img.size == 0:
                return "未知", 0.0
                
            img_tensor = self.preprocess_image(img)
            with torch.no_grad():
                output = self.models['color'](img_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)
                color = translate_color(self.models['color'].classes[pred.item()])
                return color, float(conf)
                
        except Exception as e:
            logger.error(f"颜色预测失败: {str(e)}")
            return "未知", 0.0
            
    async def analyze_frame(self, frame):
        """分析单个视频帧"""
        try:
            # 确保frame不为空
            if frame is None or frame.size == 0:
                logger.error("输入帧为空")
                return []

            # 转换为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            faces = self.detect_faces(frame_rgb)
            persons_info = []
            
            for face_box in faces:
                try:
                    x1, y1, x2, y2 = map(int, face_box[:4])
                    confidence = face_box[4]
                    
                    # 确保边界框坐标有效
                    x1, x2 = max(0, x1), min(frame.shape[1], x2)
                    y1, y2 = max(0, y1), min(frame.shape[0], y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # 提取人脸区域
                    face_img = frame_rgb[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                    
                    # 转换为PIL图像进行性别预测
                    face_pil = Image.fromarray(face_img)
                    gender, gender_conf = self.predict_gender(face_pil)
                    
                    # 年龄预测
                    age = self.predict_age(face_img)
                    
                    # 分析衣服
                    # 扩展人物区域，但确保不超出图像边界
                    body_y1 = max(0, y1 - int(face_img.shape[0] * 0.5))  # 向上扩展0.5个人脸高度
                    body_y2 = min(frame.shape[0], y2 + int(face_img.shape[0] * 2))  # 向下扩展2个人脸高度
                    body_x1 = max(0, x1 - int(face_img.shape[1] * 0.5))  # 向左扩展0.5个人脸宽度
                    body_x2 = min(frame.shape[1], x2 + int(face_img.shape[1] * 0.5))  # 向右扩展0.5个人脸宽度
                    
                    person_img = frame_rgb[body_y1:body_y2, body_x1:body_x2]
                    if person_img.size == 0:
                        continue
                        
                    clothing = self.analyze_clothing(person_img)
                    
                    person_info = {
                        'face': {
                            'detected': True,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence)
                        },
                        'gender': {
                            'detected': gender != "未知",
                            'value': gender,
                            'confidence': float(gender_conf)
                        },
                        'age': {
                            'detected': age > 0,
                            'value': float(age),
                            'confidence': 0.8  # 默认置信度
                        },
                        'clothing': {
                            'upper': {
                                'detected': clothing['upper'][0] != "未知",
                                'color': clothing['upper'][0],
                                'confidence': float(clothing['upper'][1])
                            },
                            'lower': {
                                'detected': clothing['lower'][0] != "未知",
                                'color': clothing['lower'][0],
                                'confidence': float(clothing['lower'][1])
                            }
                        }
                    }
                    
                    persons_info.append(person_info)
                    
                except Exception as e:
                    logger.error(f"处理单个人物时出错: {str(e)}")
                    continue
            
            return persons_info
            
        except Exception as e:
            logger.error(f"帧分析失败: {str(e)}")
            return []

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
                        # 直接分析当前帧
                        persons_info = await self.analyze_frame(frame)
                        
                        # 检查是否检测到人物
                        if len(persons_info) > 0:
                            # 在帧上绘制分析结果
                            result_frame = frame.copy()
                            
                            # 为每个检测到的人物绘制信息
                            for person in persons_info:
                                if person['face']['detected']:
                                    x1, y1, x2, y2 = person['face']['bbox']
                                    
                                    # 绘制人脸框
                                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # 准备文本信息
                                    text_lines = []
                                    
                                    if person['gender']['detected']:
                                        gender_text = f"性别: {person['gender']['value']} ({person['gender']['confidence']:.2f})"
                                        text_lines.append(gender_text)
                                    
                                    if person['age']['detected']:
                                        age_text = f"年龄: {person['age']['value']:.1f} ({person['age']['confidence']:.2f})"
                                        text_lines.append(age_text)
                                    
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
                            
                            # 写入标注后的帧
                            out.write(result_frame)
                            
                            # 添加帧信息
                            frame_result = {
                                'frame_number': frame_count,
                                'timestamp': frame_count / fps,
                                'num_persons': len(persons_info),
                                'persons': persons_info
                            }
                            frame_results.append(frame_result)
                        else:
                            # 如果没有检测到人物，直接写入原始帧
                            out.write(frame)
                        
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

def clear_torch_cache():
    """清理 torch hub 缓存"""
    import shutil
    from pathlib import Path
    
    cache_dir = Path.home() / '.cache/torch/hub'
    if cache_dir.exists():
        shutil.rmtree(str(cache_dir))
        logger.info("已清理 torch hub 缓存")

# 创建全局实例
video_analyzer = VideoAnalyzer() 