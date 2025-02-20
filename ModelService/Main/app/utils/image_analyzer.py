import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import os
from pathlib import Path
import numpy as np
import mediapipe as mp
from torchvision.models.segmentation import deeplabv3_resnet50
import logging
import threading
from queue import Queue
from typing import Dict, List, Optional, Tuple
import warnings
from ultralytics import YOLO
from scipy import ndimage
import base64
import json
import requests  # 替换 OpenAI 客户端，使用 requests 直接调用 API
import time
from functools import lru_cache
import hashlib
try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI import failed, enhanced analysis mode may not be available")
    OpenAI = None

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT_DIR / "model"

class Singleton(type):
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

def get_device():
    """获取可用的设备，优先使用GPU"""
    if torch.cuda.is_available():
        return ['cuda:0']
    return ['cpu']

class ModelBase:
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        raise NotImplementedError

class ColorModel(ModelBase):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.classes = None
        self.load_model()
    
    def load_model(self):
        """加载颜色分类模型"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'classes' in checkpoint:
                self.classes = checkpoint['classes']
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                raise ValueError("无效的模型检查点格式")
            
            # 创建ResNet18模型
            self.model = models.resnet18(weights=None)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential()
            self.model.fc.add_module('1', nn.Linear(num_features, 512))
            self.model.fc.add_module('2', nn.ReLU())
            self.model.fc.add_module('4', nn.Linear(512, len(self.classes)))
            
            # 加载权重
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("颜色分类模型加载成功")
        except Exception as e:
            logger.error(f"加载颜色分类模型失败: {str(e)}")
            raise
    
    def predict(self, img):
        """预测颜色类别"""
        try:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, idx = torch.max(probs, dim=1)
                return self.classes[idx.item()], conf.item()
        except Exception as e:
            logger.error(f"颜色预测失败: {str(e)}")
            return None, 0.0

class AgeModel(ModelBase):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.age_classes = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
        self.load_model()
    
    def load_model(self):
        """加载年龄估计模型"""
        try:
            class AgeEstimationModel(nn.Module):
                def __init__(self, num_classes):
                    super().__init__()
                    # 使用ResNet50作为基础模型
                    self.backbone = models.resnet50(weights=None)
                    
                    # 修改最后的全连接层，匹配训练时的结构
                    in_features = self.backbone.fc.in_features
                    self.backbone.fc = nn.Sequential(
                        nn.Linear(in_features, 1024),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1024, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(512, num_classes)
                    )
                
                def forward(self, x):
                    return self.backbone(x)
            
            # 创建模型实例
            self.model = AgeEstimationModel(num_classes=len(self.age_classes))
            
            # 加载权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 处理state_dict中的键
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # 移除'module.'前缀
                new_state_dict[k] = v
            
            # 加载处理后的权重
            self.model.load_state_dict(new_state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("年龄估计模型加载成功")
        except Exception as e:
            logger.error(f"加载年龄估计模型失败: {str(e)}")
            raise
    
    def predict(self, img):
        """预测年龄"""
        try:
            # 确保输入是PIL图像
            if not isinstance(img, Image.Image):
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    raise ValueError("输入必须是PIL图像或numpy数组")
            
            # 应用转换
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
                pred_age = self.age_classes[pred_idx.item()]
                
                # 解析年龄范围并计算中间值
                age_range = pred_age.split('-')
                if len(age_range) == 2:
                    min_age = int(age_range[0])
                    max_age = int(age_range[1])
                else:  # 处理 "71+" 这种情况
                    min_age = int(pred_age.replace('+', ''))
                    max_age = min_age + 29  # 假设最大年龄为100岁
                
                # 根据置信度在范围内插值
                predicted_age = min_age + (max_age - min_age) * confidence.item()
                
                return predicted_age, confidence.item()
        except Exception as e:
            logger.error(f"年龄预测失败: {str(e)}")
            return None, 0.0

class GenderModel(ModelBase):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.confidence_threshold = 0.5  # 降低基础阈值
        self.load_model()
        # 增强的预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """加载性别分类模型"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info("性别分类模型加载成功")
        except Exception as e:
            logger.error(f"加载性别分类模型失败: {str(e)}")
            raise

    def predict(self, img):
        """预测性别"""
        try:
            # 确保输入是PIL图像
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            # 运行YOLO预测
            results = self.model(img)
            
            if not results or len(results) == 0:
                logger.warning("性别预测没有返回结果")
                return "unknown", 0.0
            
            # 获取预测结果
            result = results[0]  # 获取第一个预测结果
            
            if not hasattr(result, 'probs') or not result.probs:
                logger.warning("性别预测概率为空")
                return "unknown", 0.0
            
            # 获取female和male的概率
            probs = result.probs
            female_prob = float(probs.data[0])
            male_prob = float(probs.data[1])
            
            # 记录原始预测概率
            logger.info(f"性别预测原始概率 - 女性: {female_prob:.4f}, 男性: {male_prob:.4f}")
            
            # 修改判断逻辑：直接选择概率最高的性别
            if female_prob >= male_prob:
                gender = 'female'
                confidence = female_prob
            else:
                gender = 'male'
                confidence = male_prob
            
            # 记录最终预测结果
            logger.info(f"性别预测结果: {gender}, 置信度: {confidence:.4f}")
            
            # 如果置信度太低，标记为unknown
            if confidence < 0.4:  # 降低阈值到0.4
                logger.warning(f"性别预测置信度过低: {confidence:.4f}")
                return "unknown", confidence
            
            return gender, confidence
            
        except Exception as e:
            logger.error(f"性别预测失败: {str(e)}")
            return "unknown", 0.0

class ImageAnalyzer(metaclass=Singleton):
    def __init__(self):
        self.device = get_device()[0]
        self.models = {}
        self.model_paths = self._get_model_paths()
        self.deeplabv3 = None
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 修改 API key 的获取逻辑
        self.api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-8ecbfb7922bc425bafb971616f5a7674')
        if self.api_key == 'sk-8ecbfb7922bc425bafb971616f5a7674':
            logger.info("使用默认 API key")
        else:
            logger.info("使用环境变量中的 API key")
        
        if OpenAI is None:
            logger.warning("OpenAI 导入失败，增强模式可能无法使用")
        
        self.load_models()
        self.cache = {}
    
    def _get_model_paths(self) -> Dict[str, str]:
        """获取模型文件路径"""
        paths = {
            'face': str(MODEL_DIR / "output/face_detection/train2/weights/best.pt"),
            'color': str(MODEL_DIR / "output/color_classification/best_model.pth"),
            'age': str(MODEL_DIR / "output/age_estimation/weights/best.pt"),
            'gender': str(MODEL_DIR / "output/gender_classification/train/weights/best.pt")
        }
        
        # 验证模型文件是否存在
        for name, path in paths.items():
            if not os.path.exists(path):
                logger.warning(f"模型文件不存在: {path}")
                paths[name] = None
        
        return paths
    
    def load_models(self):
        """加载所有模型"""
        try:
            # 加载DeepLabV3模型
            try:
                self.deeplabv3 = models.segmentation.deeplabv3_resnet50(pretrained=True)
                self.deeplabv3.eval()
                self.deeplabv3.to(self.device)
                logger.info("DeepLabV3模型加载成功")
            except Exception as e:
                logger.error(f"加载DeepLabV3模型失败: {str(e)}")
                self.deeplabv3 = None
            
            # 加载YOLO模型
            from ultralytics import YOLO
            
            # 加载人脸检测模型
            if self.model_paths['face']:
                try:
                    self.models['face'] = YOLO(self.model_paths['face'])
                    logger.info("人脸检测模型加载成功")
                except Exception as e:
                    logger.error(f"加载人脸检测模型失败: {str(e)}")
                    self.models['face'] = None
            
            # 加载颜色分类模型
            if self.model_paths['color']:
                try:
                    self.models['color'] = ColorModel(self.model_paths['color'], self.device)
                    logger.info("颜色分类模型加载成功")
                except Exception as e:
                    logger.error(f"加载颜色分类模型失败: {str(e)}")
                    self.models['color'] = None
            
            # 加载年龄估计模型
            if self.model_paths['age']:
                try:
                    self.models['age'] = AgeModel(self.model_paths['age'], self.device)
                    logger.info("年龄估计模型加载成功")
                except Exception as e:
                    logger.error(f"加载年龄估计模型失败: {str(e)}")
                    self.models['age'] = None
            
            # 加载性别分类模型
            if self.model_paths['gender']:
                try:
                    self.models['gender'] = GenderModel(self.model_paths['gender'], self.device)
                    logger.info("性别分类模型加载成功")
                except Exception as e:
                    logger.error(f"加载性别分类模型失败: {str(e)}")
                    self.models['gender'] = None
            
        except ImportError as e:
            logger.error(f"加载模型失败: {str(e)}")
    
    def _analyze_clothing_color(self, img, segmentation_mask, person_bbox):
        """分析特定人物的上衣和下衣颜色"""
        try:
            # 检查输入
            if img is None or len(img.shape) != 3:
                logger.warning("无效的图像输入")
                return {
                    "upper": {"color": "unknown", "confidence": 0.0},
                    "lower": {"color": "unknown", "confidence": 0.0}
                }
            
            # 获取人物区域
            if person_bbox is None or len(person_bbox) != 4:
                logger.warning("无效的人物边界框")
                return {
                    "upper": {"color": "unknown", "confidence": 0.0},
                    "lower": {"color": "unknown", "confidence": 0.0}
                }
            
            x1, y1, x2, y2 = person_bbox
            face_height = y2 - y1
            face_width = x2 - x1
            img_height, img_width = img.shape[:2]
            
            clothing_colors = {
                "upper": {"color": "unknown", "confidence": 0.0},
                "lower": {"color": "unknown", "confidence": 0.0}
            }

            def check_overlap(region, other_boxes):
                """检查区域是否与其他人物重叠"""
                rx1, ry1, rx2, ry2 = region
                for box in other_boxes:
                    if box is None or len(box) != 4:
                        continue
                    bx1, by1, bx2, by2 = box
                    # 计算重叠区域
                    x_left = max(rx1, bx1)
                    y_top = max(ry1, by1)
                    x_right = min(rx2, bx2)
                    y_bottom = min(ry2, by2)
                    
                    if x_right > x_left and y_bottom > y_top:
                        overlap_area = (x_right - x_left) * (y_bottom - y_top)
                        region_area = (rx2 - rx1) * (ry2 - ry1)
                        overlap_ratio = overlap_area / region_area
                        if overlap_ratio > 0.3:  # 如果重叠面积超过30%
                            return True
                return False
            
            def analyze_color_region(img_region, region_type="upper"):
                """分析特定区域的颜色"""
                if img_region is None or img_region.size == 0:
                    return "unknown", 0.0
                
                try:
                    if self.models.get('color'):
                        rgb_img = cv2.cvtColor(img_region, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_img)
                        
                        # 基本颜色映射和阈值
                        basic_colors = {
                            'black': ([0, 0, 0], 50),        # 黑色阈值
                            'white': ([255, 255, 255], 50),  # 白色阈值
                            'red': ([255, 0, 0], 80),        # 红色阈值
                            'green': ([0, 255, 0], 80),      # 绿色阈值
                            'blue': ([0, 0, 255], 80),       # 蓝色阈值
                            'yellow': ([255, 255, 0], 80),   # 黄色阈值
                            'gray': ([128, 128, 128], 50)    # 灰色阈值
                        }
                        
                        # 1. 首先尝试使用模型进行预测
                        colors = []
                        confidences = []
                        scales = [0.8, 1.0, 1.2]
                        angles = [0]  # 简化角度以提高性能
                        
                        for scale in scales:
                            scaled_img = pil_img.resize((int(224*scale), int(224*scale)))
                            for angle in angles:
                                try:
                                    if angle != 0:
                                        rotated_img = scaled_img.rotate(angle, expand=True)
                                    else:
                                        rotated_img = scaled_img
                                    
                                    color, conf = self.models['color'].predict(rotated_img)
                                    if color:
                                        colors.append(color)
                                        confidences.append(conf)
                                except Exception as e:
                                    continue
                        
                        # 2. 如果模型预测有结果且置信度足够
                        if colors:
                            from collections import Counter
                            color_counts = Counter(colors)
                            most_common_color = color_counts.most_common(1)[0][0]
                            color_confidences = [conf for c, conf in zip(colors, confidences) if c == most_common_color]
                            avg_conf = sum(color_confidences) / len(color_confidences)
                            
                            if avg_conf > 0.3:  # 降低置信度阈值到0.3
                                return most_common_color, avg_conf
                        
                        # 3. 如果模型预测不可靠，使用基本颜色分析
                        # 计算图像的平均颜色
                        avg_color = cv2.mean(rgb_img)[:3]
                        hsv_img = cv2.cvtColor(img_region, cv2.COLOR_BGR2HSV)
                        avg_hsv = cv2.mean(hsv_img)[:3]
                        
                        # 计算亮度和饱和度
                        brightness = avg_hsv[2]  # V通道
                        saturation = avg_hsv[1]  # S通道
                        
                        # 根据HSV值判断基本颜色
                        if brightness < 50:  # 暗色
                            return 'black', 0.7
                        elif brightness > 200 and saturation < 50:  # 亮色且低饱和度
                            return 'white', 0.7
                        elif saturation < 30:  # 低饱和度
                            return 'gray', 0.6
                        else:
                            # 计算与基本颜色的距离
                            min_dist = float('inf')
                            selected_color = 'unknown'
                            max_confidence = 0.0
                            
                            for color_name, (rgb, threshold) in basic_colors.items():
                                dist = sum((a - b) ** 2 for a, b in zip(avg_color, rgb)) ** 0.5
                                if dist < min_dist:
                                    min_dist = dist
                                    confidence = max(0.4, 1 - (dist / 255))  # 确保至少有0.4的置信度
                                    selected_color = color_name
                                    max_confidence = confidence
                            
                            return selected_color, max_confidence
                    
                except Exception as e:
                    logger.error(f"{region_type} - 颜色分析失败: {str(e)}")
                    # 即使发生错误，也尝试返回基本颜色
                    try:
                        avg_color = cv2.mean(img_region)[:3]
                        if sum(avg_color) < 380:
                            return 'black', 0.4
                        elif sum(avg_color) > 650:
                            return 'white', 0.4
                        return 'gray', 0.4
                    except:
                        return 'unknown', 0.0
            
            # 获取其他人物的边界框
            other_boxes = []
            if hasattr(self, 'current_boxes'):
                other_boxes = [box for box in self.current_boxes if box != person_bbox]
            
            # 分析上衣颜色
            upper_region = [
                max(0, int((x1 + x2) / 2 - face_width * 1.5)),
                min(img_height - 1, y2),
                min(img_width, int((x1 + x2) / 2 + face_width * 1.5)),
                min(img_height, int(y2 + face_height * 2.0))
            ]
            
            if not check_overlap(upper_region, other_boxes):
                upper_img = img[upper_region[1]:upper_region[3], upper_region[0]:upper_region[2]].copy()
                color, conf = analyze_color_region(upper_img, "upper")
                clothing_colors["upper"] = {"color": color, "confidence": conf}
            else:
                clothing_colors["upper"] = {"color": "overlapped", "confidence": 0.0}
            
            # 分析下衣颜色
            lower_region = [
                max(0, int((x1 + x2) / 2 - face_width * 1.5)),
                min(img_height - 1, upper_region[3]),
                min(img_width, int((x1 + x2) / 2 + face_width * 1.5)),
                min(img_height, int(upper_region[3] + face_height * 1.5))
            ]
            
            if not check_overlap(lower_region, other_boxes):
                lower_img = img[lower_region[1]:lower_region[3], lower_region[0]:lower_region[2]].copy()
                color, conf = analyze_color_region(lower_img, "lower")
                clothing_colors["lower"] = {"color": color, "confidence": conf}
            else:
                clothing_colors["lower"] = {"color": "overlapped", "confidence": 0.0}
            
            # 确保返回值不为 unknown
            for part in ["upper", "lower"]:
                if clothing_colors[part]["color"] == "unknown":
                    # 如果是 unknown，使用基本的颜色分析
                    region_img = upper_img if part == "upper" else lower_img
                    if region_img is not None and region_img.size > 0:
                        avg_color = cv2.mean(region_img)[:3]
                        if sum(avg_color) < 380:
                            clothing_colors[part] = {"color": "black", "confidence": 0.4}
                        elif sum(avg_color) > 650:
                            clothing_colors[part] = {"color": "white", "confidence": 0.4}
                        else:
                            clothing_colors[part] = {"color": "gray", "confidence": 0.4}
            
            logger.info(f"颜色分析结果: {clothing_colors}")
            return clothing_colors
        
        except Exception as e:
            logger.error(f"衣服颜色分析失败: {str(e)}")
            return {
                "upper": {"color": "gray", "confidence": 0.3},  # 使用默认颜色而不是 unknown
                "lower": {"color": "gray", "confidence": 0.3}
            }
    
    def _prepare_face_image(self, face_img):
        """准备人脸图像用于模型输入"""
        try:
            # 转换为RGB
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                return None
            
            # 调整大小
            face_rgb = cv2.resize(face_rgb, (224, 224))
            
            # 转换为PIL图像
            face_pil = Image.fromarray(face_rgb)
            return face_pil
        except Exception as e:
            logger.error(f"人脸图像预处理失败: {str(e)}")
            return None
    
    def analyze_with_qwen(self, image_path: str) -> Dict:
        """使用Qwen-VL模型分析图片"""
        try:
            if OpenAI is None:
                logger.error("OpenAI 模块未正确加载")
                return {"error": "增强分析模式不可用"}

            # 创建 OpenAI 客户端
            try:
                client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                logger.info("成功创建 OpenAI 客户端")
            except Exception as e:
                logger.error(f"创建 OpenAI 客户端失败: {str(e)}")
                return {"error": "创建 OpenAI 客户端失败"}

            # 读取并编码图片
            try:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                logger.info("成功读取并编码图片")
            except Exception as e:
                logger.error(f"图片编码失败: {str(e)}")
                return {"error": "图片编码失败"}

            try:
                # 发送请求
                messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "你是一个图像分析助手。"}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                                },
                                {
                                    "type": "text",
                                    "text": """请分析图像中的人物信息，并按照以下JSON格式返回结果：
{
    "detected": 人物数量,
    "persons": [
        {
            "gender": "male/female",
            "gender_confidence": 0.95,
            "age": 25,
            "age_confidence": 0.85,
            "upper_color": "red/blue/green/...",
            "upper_color_confidence": 0.8,
            "lower_color": "red/blue/green/...",
            "lower_color_confidence": 0.8,
            "bbox": [x1, y1, x2, y2]
        }
    ],
    "success": true
}"""
                            }
                        ]
                    }
                ]
                
                logger.info("发送 Qwen-VL 请求...")
                completion = client.chat.completions.create(
                    model="qwen-vl-max-latest",
                    messages=messages
                )
                
                # 打印完整的 API 响应
                logger.info(f"Qwen-VL API 完整响应: {completion}")

                # 解析结果
                result_text = completion.choices[0].message.content
                logger.info(f"Qwen-VL 返回原始结果: {result_text}")

                # 提取 JSON 部分
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1

                if json_start >= 0 and json_end > json_start:
                    try:
                        json_text = result_text[json_start:json_end]
                        json_text = json_text.replace('\n', ' ').replace('\r', ' ')
                        json_text = ' '.join(json_text.split())
                        parsed_result = json.loads(json_text)
                        logger.info(f"成功解析 Qwen-VL 返回结果: {json.dumps(parsed_result, ensure_ascii=False, indent=2)}")
                        
                        # 确保结果格式统一
                        standardized_result = {
                            "detected": parsed_result.get("detected", 0),
                            "persons": [],
                            "success": True
                        }
                        
                        for person in parsed_result.get("persons", []):
                            standardized_person = {
                                "gender": person.get("gender", "unknown"),
                                "gender_confidence": person.get("gender_confidence", 0.0),
                                "age": person.get("age", 0),
                                "age_confidence": person.get("age_confidence", 0.0),
                                "upper_color": person.get("upper_color", "unknown"),
                                "upper_color_confidence": person.get("upper_color_confidence", 0.0),
                                "lower_color": person.get("lower_color", "unknown"),
                                "lower_color_confidence": person.get("lower_color_confidence", 0.0),
                                "bbox": person.get("bbox", [0, 0, 0, 0])
                            }
                            standardized_result["persons"].append(standardized_person)
                        
                        logger.info(f"标准化后的结果: {json.dumps(standardized_result, ensure_ascii=False, indent=2)}")
                        return standardized_result
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析失败: {str(e)}")
                        return {"error": "JSON解析失败", "raw_text": result_text}
                else:
                    logger.error(f"无法从响应中提取JSON结果: {result_text}")
                    return {"error": "无法解析返回结果", "raw_text": result_text}

            except Exception as e:
                logger.error(f"Qwen-VL API调用失败: {str(e)}")
                return {"error": f"API调用失败: {str(e)}"}

        except Exception as e:
            logger.error(f"Qwen-VL分析失败: {str(e)}")
            return {"error": str(e)}

    def merge_results(self, local_result: Dict, qwen_result: Dict, local_weight: float = 0.01, qwen_weight: float = 0.99) -> Dict:
        """合并本地模型和Qwen-VL的分析结果"""
        try:
            logger.info("开始合并分析结果")
            logger.info(f"权重设置 - 本地模型: {local_weight}, Qwen-VL: {qwen_weight}")
            
            # 验证权重和
            total_weight = local_weight + qwen_weight
            if abs(total_weight - 1.0) > 0.0001:
                logger.warning(f"权重和不为1，进行归一化处理: {total_weight}")
                local_weight = local_weight / total_weight
                qwen_weight = qwen_weight / total_weight
            
            merged_result = {
                "detected": 0,
                "persons": [],
                "success": True
            }
            
            # 验证输入数据
            if not isinstance(local_result, dict) or not isinstance(qwen_result, dict):
                logger.error("输入结果格式无效")
                return {
                    "detected": 0,
                    "persons": [],
                    "success": False,
                    "error": "Invalid input format"
                }
            
            # 获取检测到的人数
            local_persons = local_result.get("persons", [])
            qwen_persons = qwen_result.get("persons", [])
            
            logger.info(f"本地模型检测到 {len(local_persons)} 人")
            logger.info(f"Qwen-VL检测到 {len(qwen_persons)} 人")
            
            # 使用 Qwen-VL 的检测结果作为主要结果
            merged_result["detected"] = len(qwen_persons) or len(local_persons)
            
            # 对每个检测到的人物进行结果合并
            for i in range(merged_result["detected"]):
                qwen_person = qwen_persons[i] if i < len(qwen_persons) else None
                local_person = local_persons[i] if i < len(local_persons) else None
                
                # 如果 Qwen-VL 有结果，优先使用
                if qwen_person:
                    person_data = {
                        "gender": qwen_person.get("gender", "unknown"),
                        "gender_confidence": qwen_person.get("gender_confidence", 0.0) * qwen_weight,
                        "age": qwen_person.get("age", 0),
                        "age_confidence": qwen_person.get("age_confidence", 0.0) * qwen_weight,
                        "upper_color": qwen_person.get("upper_color", "unknown"),
                        "upper_color_confidence": qwen_person.get("upper_color_confidence", 0.0) * qwen_weight,
                        "lower_color": qwen_person.get("lower_color", "unknown"),
                        "lower_color_confidence": qwen_person.get("lower_color_confidence", 0.0) * qwen_weight,
                        "bbox": qwen_person.get("bbox", [0, 0, 0, 0])
                    }
                # 如果只有本地模型结果
                elif local_person:
                    person_data = {
                        "gender": local_person.get("gender", "unknown"),
                        "gender_confidence": local_person.get("gender_confidence", 0.0) * local_weight,
                        "age": local_person.get("age", 0),
                        "age_confidence": local_person.get("age_confidence", 0.0) * local_weight,
                        "upper_color": local_person.get("upper_color", "unknown"),
                        "upper_color_confidence": local_person.get("upper_color_confidence", 0.0) * local_weight,
                        "lower_color": local_person.get("lower_color", "unknown"),
                        "lower_color_confidence": local_person.get("lower_color_confidence", 0.0) * local_weight,
                        "bbox": local_person.get("bbox", [0, 0, 0, 0])
                    }
                
                merged_result["persons"].append(person_data)
                logger.info(f"人物 {i+1} 合并结果: {json.dumps(person_data, ensure_ascii=False, indent=2)}")

            return merged_result

        except Exception as e:
            logger.error(f"合并结果时出错: {str(e)}")
            return {
                "detected": 0,
                "persons": [],
                "success": False,
                "error": str(e)
            }

    def _get_image_hash(self, image_path: str) -> str:
        """计算图片哈希值"""
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def analyze_image(self, image_path: str, mode: str = "normal") -> Dict:
        """分析图像"""
        try:
            logger.info(f"开始图像分析，模式: {mode}")
            start_time = time.time()
            
            # 使用本地模型进行分析
            local_result = self._analyze_with_local_models(image_path)
            logger.info(f"本地模型分析结果: {json.dumps(local_result, ensure_ascii=False, indent=2)}")
            
            # 在增强模式下，同时使用本地模型和 Qwen-VL
            if mode == "enhanced":
                logger.info("使用增强模式，调用 Qwen-VL API")
                try:
                    # 使用 Qwen-VL 进行分析
                    qwen_result = self.analyze_with_qwen(image_path)
                    logger.info(f"Qwen-VL 分析结果: {json.dumps(qwen_result, ensure_ascii=False, indent=2)}")
                    
                    # 检查 Qwen-VL 结果
                    if "error" not in qwen_result:
                        logger.info("成功获取 Qwen-VL 分析结果，开始合并结果")
                        
                        # 确保 local_result 中有必要的字段
                        if "persons" not in local_result:
                            local_result["persons"] = []
                        if "detected" not in local_result:
                            local_result["detected"] = 0
                        
                        # 合并两个模型的结果
                        try:
                            final_result = self.merge_results(
                                local_result,
                                qwen_result,
                                local_weight=0.01,  # 修改权重为 1%
                                qwen_weight=0.99    # 修改权重为 99%
                            )
                            logger.info(f"合并后的最终结果: {json.dumps(final_result, ensure_ascii=False, indent=2)}")
                        except Exception as e:
                            logger.error(f"合并结果失败: {str(e)}")
                            final_result = local_result
                    else:
                        logger.warning(f"Qwen-VL分析失败，使用本地模型结果: {qwen_result.get('error')}")
                        final_result = local_result
                except Exception as e:
                    logger.error(f"Qwen-VL分析失败: {str(e)}")
                    final_result = local_result
            else:
                logger.info("使用普通模式，仅使用本地模型")
                final_result = local_result
            
            # 确保结果包含所有必要字段
            if "persons" not in final_result:
                final_result["persons"] = []
            if "detected" not in final_result:
                final_result["detected"] = len(final_result.get("persons", []))
            
            # 添加处理时间和模式信息
            final_result["processing_time"] = time.time() - start_time
            final_result["mode"] = mode
            
            logger.info(f"图像分析完成，耗时: {final_result['processing_time']:.2f}秒")
            return final_result
            
        except Exception as e:
            logger.error(f"图像分析失败: {str(e)}")
            return {
                "error": str(e),
                "detected": 0,
                "persons": [],
                "mode": mode,
                "processing_time": time.time() - start_time
            }

    def _analyze_with_local_models(self, image_path: str) -> Dict:
        """使用本地模型进行分析"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 获取图像尺寸
            img_height, img_width = img.shape[:2]
            
            # 初始化结果
            result = {
                "detected": 0,
                "persons": [],
                "success": True
            }
            
            # 人脸检测
            if self.models.get('face'):
                try:
                    # 使用YOLO进行人脸检测
                    face_results = self.models['face'](img)
                    
                    if len(face_results) > 0 and len(face_results[0].boxes) > 0:
                        boxes = face_results[0].boxes
                        self.current_boxes = [box.xyxy[0].tolist() for box in boxes]
                        for idx, box in enumerate(boxes):
                            try:
                                # 获取边界框坐标
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                confidence = float(box.conf[0])
                                
                                # 提取人脸区域
                                face_img = img[y1:y2, x1:x2]
                                if face_img.size == 0:
                                    continue
                                
                                # 转换为PIL图像
                                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                                
                                # 性别预测
                                gender, gender_conf = self.models['gender'].predict(face_img)
                                
                                # 年龄预测
                                age, age_conf = self.models['age'].predict(face_pil)
                                
                                # 衣服颜色分析
                                clothing_colors = self._analyze_clothing_color(img, None, [x1, y1, x2, y2])
                                
                                # 收集人物信息
                                person_data = {
                                    "gender": gender,
                                    "gender_confidence": gender_conf,
                                    "age": age,
                                    "age_confidence": age_conf,
                                    "upper_color": clothing_colors.get("upper", {}).get("color", "unknown"),
                                    "upper_color_confidence": clothing_colors.get("upper", {}).get("confidence", 0.0),
                                    "lower_color": clothing_colors.get("lower", {}).get("color", "unknown"),
                                    "lower_color_confidence": clothing_colors.get("lower", {}).get("confidence", 0.0),
                                    "bbox": [x1, y1, x2, y2]
                                }
                                
                                result["persons"].append(person_data)
                            
                            except Exception as e:
                                logger.error(f"处理人物 {idx} 时出错: {str(e)}")
                                continue
                    
                    result["detected"] = len(result["persons"])
                
                except Exception as e:
                    logger.error(f"人脸检测失败: {str(e)}")
            
            logger.info(f"本地模型分析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return result
        
        except Exception as e:
            logger.error(f"本地模型分析失败: {str(e)}")
            return {
                "detected": 0,
                "persons": [],
                "success": False,
                "error": str(e)
            }

# 创建全局分析器实例
image_analyzer = ImageAnalyzer() 