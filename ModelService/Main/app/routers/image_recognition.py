import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import os
from pathlib import Path
import random
from ultralytics import YOLO
from ultralytics.models.yolo.model import YOLO as YOLOModel
from ultralytics.nn.tasks import DetectionModel
import numpy as np
import threading
from queue import Queue, Empty
from typing import Optional
import mediapipe as mp
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制 TensorFlow 警告
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import base64
from datetime import datetime
from app.utils.color_mapping import translate_color
import logging
from app.utils.image_analyzer import image_analyzer
import requests
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """获取可用的设备，优先使用所有可用的GPU"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            print(f"发现 {num_gpus} 个 GPU:")
            devices = []
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
                devices.append(f'cuda:{i}')
            return devices
    print("未找到 GPU，使用 CPU")
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
    
    def _load_state_dict(self, checkpoint):
        """统一的模型状态加载方法"""
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
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
            
        return new_state_dict

class ColorModel(ModelBase, metaclass=Singleton):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.classes = None
        self.load_model()
    
    def load_model(self):
        class ColorClassifier(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.model = models.resnet18(weights=None)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Sequential()
                self.model.fc.add_module('1', nn.Linear(num_features, 512))
                self.model.fc.add_module('2', nn.ReLU())
                self.model.fc.add_module('4', nn.Linear(512, num_classes))
            
            def forward(self, x):
                return self.model(x)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        num_classes = len(checkpoint['classes'])
        self.model = ColorClassifier(num_classes)
        state_dict = self._load_state_dict(checkpoint)
        self.model.load_state_dict(state_dict)
        self.classes = checkpoint['classes']
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, img):
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, idx = torch.max(probs, dim=1)
            return self.classes[idx.item()], conf.item()

class FaceModel(ModelBase, metaclass=Singleton):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.conf = 0.25  # 置信度阈值
        self.model = None
        self.img_size = 640  # YOLO 默认输入尺寸
        print(f"初始化人脸检测模型，路径: {model_path}, 设备: {device}")
        self.load_model()
    
    def load_model(self):
        """加载人脸检测模型"""
        try:
            # 确保权重文件存在
            weights_path = Path(self.model_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"错误：模型权重不存在: {weights_path}")
            
            print(f"开始加载人脸检测模型: {weights_path}")
            
            # 加载模型
            try:
                self.model = YOLO(str(weights_path))
                print("YOLO模型加载成功")
            except Exception as e:
                print(f"YOLO模型加载失败: {str(e)}")
                raise
            
            # 设置设备
            try:
                if self.device == 'auto':
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                else:
                    device = self.device
                
                self.model.to(device)
                print(f"模型已成功移至设备: {device}")
            except Exception as e:
                print(f"设置模型设备失败: {str(e)}")
                raise
            
            print("人脸检测模型加载完成")
            
        except Exception as e:
            error_msg = f"加载人脸检测模型失败: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
    def preprocess_image(self, img):
        """预处理图像"""
        try:
            # 转换为RGB格式（如果是BGR）
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            # 保存原始尺寸
            orig_h, orig_w = img_rgb.shape[:2]
            
            # 计算缩放比例，保持宽高比
            scale = min(self.img_size / orig_h, self.img_size / orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            
            # 调整图像大小
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 创建目标尺寸的画布
            canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
            # 计算偏移量，确保图像居中
            offset_h = (self.img_size - new_h) // 2
            offset_w = (self.img_size - new_w) // 2
            
            # 将调整后的图像放在画布中央
            canvas[offset_h:offset_h + new_h, offset_w:offset_w + new_w] = img_resized
            
            # 转换为 PyTorch tensor，并添加批次维度
            img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            return img_tensor, scale, (offset_h, offset_w)
            
        except Exception as e:
            print(f"图像预处理失败: {str(e)}")
            raise
    
    def predict(self, img):
        """预测人脸"""
        try:
            if self.model is None:
                raise RuntimeError("模型未正确加载")
            
            # 预处理图像
            processed_img, scale, (offset_h, offset_w) = self.preprocess_image(img)
            
            # 使用predict方法
            results = self.model.predict(
                source=processed_img,
                conf=self.conf,
                verbose=False
            )
            
            # 如果有检测结果，调整边界框坐标
            if len(results) > 0 and len(results[0].boxes) > 0:
                result = results[0]
                boxes = result.boxes
                # 创建新的边界框列表
                adjusted_boxes = []
                
                for box in boxes:
                    # 获取原始坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # 还原到原始图像坐标
                    x1 = int((x1 - offset_w) / scale)
                    y1 = int((y1 - offset_h) / scale)
                    x2 = int((x2 - offset_w) / scale)
                    y2 = int((y2 - offset_h) / scale)
                    
                    # 创建新的检测框数据
                    new_box = torch.tensor([[x1, y1, x2, y2]], device=box.xyxy.device)
                    if hasattr(box, 'conf'):
                        conf = box.conf
                        new_box = torch.cat([new_box, conf.unsqueeze(-1)], dim=1)
                    if hasattr(box, 'cls'):
                        cls = box.cls
                        new_box = torch.cat([new_box, cls.unsqueeze(-1)], dim=1)
                    
                    # 使用新的坐标创建Boxes对象
                    from ultralytics.engine.results import Boxes
                    adjusted_box = Boxes(new_box, result.orig_shape)
                    adjusted_boxes.append(adjusted_box)
                
                # 更新结果中的boxes
                result.boxes = Boxes(
                    torch.cat([box.data for box in adjusted_boxes]),
                    result.orig_shape
                )
            
            return results
            
        except Exception as e:
            print(f"人脸检测预测失败: {str(e)}")
            return []

class AgeModel(ModelBase, metaclass=Singleton):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.age_classes = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']
        self.load_model()
    
    def load_model(self):
        class AgeEstimationModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # 使用ResNet50作为基础模型
                self.backbone = models.resnet50(weights=None)
                
                # 修改最后的全连接层，匹配训练的结构
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
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = AgeEstimationModel(num_classes=len(self.age_classes))
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # 处理DataParallel的state_dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # 移除'module.'前缀
                name = k[7:]  # module.xxx -> xxx
            else:
                name = k
            new_state_dict[name] = v
            
        # 加载处理后的权重
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, img):
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
            
            return predicted_age

class GenderModel(ModelBase, metaclass=Singleton):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.confidence_threshold = 0.7
        self.load_model()
        # 定义转换
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # This will normalize to 0-1
        ])
    
    def load_model(self):
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
    
    def predict(self, img):
        """使用YOLO模型进行性别分类"""
        try:
            # 确保输入是numpy数组
            if not isinstance(img, np.ndarray):
                raise ValueError("输入必须是numpy数组")
            
            # 确保图像是BGR格式（OpenCV默认格式）
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise ValueError("输入图像必须是3通道BGR格式")
            
            # 转换为RGB格式
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 应用转换并确保值在0-1范围内
            img_tensor = self.transform(img_rgb)
            
            # 使用predict方法
            results = self.model.predict(source=img_tensor, verbose=False)
            
            if len(results) > 0:
                result = results[0]
                if hasattr(result, 'probs') and result.probs is not None:
                    # 获取类别索引和置信度
                    gender_idx = int(result.probs.top1)
                    confidence = float(result.probs.top1conf)
                    
                    # 应用置信度阈值
                    if confidence < self.confidence_threshold:
                        return "未知", confidence
                    
                    return "男" if gender_idx == 0 else "女", confidence
            
            return "未知", 0.0
            
        except Exception as e:
            print(f"性别预测失败: {str(e)}")
            return "未知", 0.0

class ClothingDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # 加载语义分割模型
        self.segmentation_model = None
        try:
            # 更新模型路径
            weights_path = Path("E:/MyUser/GraduationDesign/Programe/ModelServicepro/ModelService/Main/model/z_model_use/model/deeplabv3_resnet50.pth")
            
            if not weights_path.exists():
                print(f"提示: 未找到DeepLabV3模型文件: {weights_path}")
            else:
                self.segmentation_model = deeplabv3_resnet50(weights=None)
                state_dict = torch.load(weights_path, map_location='cpu')
                # 过滤掉aux_classifier相关的键
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                    if not k.startswith('aux_classifier')}
                # 使用strict=False允许加载不完全匹配的权重
                self.segmentation_model.load_state_dict(filtered_state_dict, strict=False)
                if torch.cuda.is_available():
                    self.segmentation_model = self.segmentation_model.cuda()
                self.segmentation_model.eval()  # 确保模型处于评估模式
                print("成功加载分割模型")
                
                # 添加图像预处理转换
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((640, 640), antialias=True),  # 调整到模型需要的尺寸
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                ])
        except Exception as e:
            print(f"加载分割模型出错: {str(e)}")
            print("将使用简单的区域检测方法")
            self.segmentation_model = None
    
    def preprocess_image(self, img):
        """预处理图像用于分割模型"""
        try:
            if isinstance(img, np.ndarray):
                # 确保图像是RGB格式
                if len(img.shape) == 3 and img.shape[2] == 3:
                    if img.dtype == np.uint8:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("输入图像必须是3通道RGB格式")
            
            # 应用转换
            img_tensor = self.transform(img)
            # 添加批次维度
            img_tensor = img_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            return img_tensor
        except Exception as e:
            print(f"图像预处理失败: {str(e)}")
            raise
    
    def detect_clothing_regions(self, img, face_box):
        """检测上衣和下装区域"""
        x1, y1, x2, y2 = face_box
        face_height = y2 - y1
        face_width = x2 - x1
        h, w = img.shape[:2]
        
        # 使用简单的几何方法估计衣服区域
        # 上衣区域：从脸部底部开始，向下延伸1.5倍人脸高度
        upper_y1 = y2
        upper_y2 = min(upper_y1 + int(face_height * 1.5), h)
        upper_x1 = max(0, x1 - int(face_width * 0.3))
        upper_x2 = min(x2 + int(face_width * 0.3), w)
        
        # 下装区域：从上衣底部开始，向下延伸2倍人脸高度
        lower_y1 = upper_y2
        lower_y2 = min(lower_y1 + int(face_height * 2.0), h)
        lower_x1 = max(0, x1 - int(face_width * 0.5))
        lower_x2 = min(x2 + int(face_width * 0.5), w)
        
        # 如果有分割模型，尝试优化区域
        if self.segmentation_model is not None:
            try:
                with torch.no_grad():
                    # 预处理图像
                    img_tensor = self.preprocess_image(img)
                    # 运行模型
                    output = self.segmentation_model(img_tensor)['out'][0]
                    # 获取预测结果
                    segmentation = torch.argmax(output, dim=0).cpu().numpy()
                    
                    # 调整分割图大小以匹配原始图像
                    segmentation = cv2.resize(segmentation.astype(np.float32), 
                                           (w, h), 
                                           interpolation=cv2.INTER_NEAREST)
                    
                    # 使用分割结果优化区域
                    person_mask = (segmentation == 15)  # COCO中的人类类别
                    
                    # 优化上衣区域
                    upper_mask = person_mask[upper_y1:upper_y2, upper_x1:upper_x2]
                    if upper_mask.any():
                        rows = np.any(upper_mask, axis=1)
                        cols = np.any(upper_mask, axis=0)
                        y_indices = np.where(rows)[0]
                        x_indices = np.where(cols)[0]
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            upper_y1 += y_indices[0]
                            upper_y2 = upper_y1 + y_indices[-1]
                            upper_x1 += x_indices[0]
                            upper_x2 = upper_x1 + x_indices[-1]
                    
                    # 优化下装区域
                    lower_mask = person_mask[lower_y1:lower_y2, lower_x1:lower_x2]
                    if lower_mask.any():
                        rows = np.any(lower_mask, axis=1)
                        cols = np.any(lower_mask, axis=0)
                        y_indices = np.where(rows)[0]
                        x_indices = np.where(cols)[0]
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            lower_y1 += y_indices[0]
                            lower_y2 = lower_y1 + y_indices[-1]
                            lower_x1 += x_indices[0]
                            lower_x2 = lower_x1 + x_indices[-1]
                            
            except Exception as e:
                print(f"使用分割模型优化区域时出错: {str(e)}")
        
        # 提取区域
        upper_region = img[upper_y1:upper_y2, upper_x1:upper_x2]
        lower_region = img[lower_y1:lower_y2, lower_x1:lower_x2]
        
        return {
            'upper': {
                'region': upper_region,
                'bbox': (upper_x1, upper_y1, upper_x2, upper_y2)
            },
            'lower': {
                'region': lower_region,
                'bbox': (lower_x1, lower_y1, lower_x2, lower_y2)
            }
        }

class ImageProcessor:
    def __init__(self, devices):
        self.devices = devices
        self.current_device_idx = 0
        self.models = {}
        self.result_queue = Queue()
        self.clothing_detector = ClothingDetector()
        self.load_models()
    
    def get_next_device(self):
        """轮询方式获取下一个设备"""
        device = self.devices[self.current_device_idx]
        self.current_device_idx = (self.current_device_idx + 1) % len(self.devices)
        return device
    
    def load_models(self):
        """加载所有模型"""
        # 获取项目根目录
        ROOT_DIR = Path(__file__).parent.parent.parent
        MODEL_DIR = ROOT_DIR / "model/output"
        
        # 检查模型目录是否存在
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"模型目录不存在: {MODEL_DIR}")
        
        model_configs = {
            'face': {
                'path': str(MODEL_DIR / 'face_detection/train2/weights/best.pt'),
                'class': FaceModel
            },
            'color': {
                'path': str(MODEL_DIR / 'color_classification/best_model.pth'),
                'class': ColorModel
            },
            'age': {
                'path': str(MODEL_DIR / 'age_estimation/weights/best.pt'),
                'class': AgeModel
            },
            'gender': {
                'path': str(MODEL_DIR / 'gender_classification/train/weights/best.pt'),
                'class': GenderModel
            }
        }
        
        # 打印模型路径信息
        print("\n模型路径配置:")
        for model_name, config in model_configs.items():
            print(f"{model_name}: {config['path']}")
            if Path(config['path']).exists():
                print(f"  ✓ 文件存在")
            else:
                print(f"  × 文件不存在")
        print()
        
        load_errors = []
        for model_name, config in model_configs.items():
            try:
                device = self.get_next_device()
                print(f"正在加载 {model_name} 模型到设备 {device}...")
                
                # 检查模型文件是否存在
                model_path = Path(config['path'])
                if not model_path.exists():
                    raise FileNotFoundError(f"模型文件不存在: {model_path}")
                
                # 确保父目录存在
                model_path.parent.mkdir(parents=True, exist_ok=True)
                
                self.models[model_name] = config['class'](str(model_path), device)
                print(f"{model_name} 模型加载成功")
            except Exception as e:
                error_msg = f"{model_name} 模型加载失败: {str(e)}"
                print(error_msg)
                load_errors.append(error_msg)
        
        if load_errors:
            raise RuntimeError("部分模型加载失败:\n" + "\n".join(load_errors))
    
    def process_image(self, img_path: Path):
        try:
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"无法读取图片: {img_path}")
            
            # 创建结果图片
            result_img = img.copy()
            
            # 首先进行人脸检测
            face_results = self.models['face'].predict(img)
            
            # 处理检测到的每个人脸
            faces_info = []
            if len(face_results) > 0 and len(face_results[0].boxes) > 0:
                for box in face_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 确保坐标在图像范围内
                    x1, x2 = max(0, x1), min(img.shape[1], x2)
                    y1, y2 = max(0, y1), min(img.shape[0], y2)
                    
                    # 提取人脸区域
                    face_img = img[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                    
                    # 检测衣服区域
                    clothing_regions = self.clothing_detector.detect_clothing_regions(img, (x1, y1, x2, y2))
                    if clothing_regions is None:
                        continue
                    
                    # 转换为RGB格式
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    
                    # 预测年龄和性别
                    predicted_age = self.models['age'].predict(face_pil)
                    predicted_gender, gender_conf = self.models['gender'].predict(face_img)
                    
                    # 预测上衣和下装颜色
                    upper_img = clothing_regions['upper']['region']
                    lower_img = clothing_regions['lower']['region']
                    
                    if upper_img.size > 0:
                        upper_rgb = cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB)
                        upper_pil = Image.fromarray(upper_rgb)
                        upper_color, upper_conf = self.models['color'].predict(upper_pil)
                    else:
                        upper_color, upper_conf = "未知", 0.0
                    
                    if lower_img.size > 0:
                        lower_rgb = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)
                        lower_pil = Image.fromarray(lower_rgb)
                        lower_color, lower_conf = self.models['color'].predict(lower_pil)
                    else:
                        lower_color, lower_conf = "未知", 0.0
                    
                    # 在图片上绘制结果
                    color = (0, 255, 0) if predicted_gender != "未知" else (0, 165, 255)
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)  # 人脸框
                    
                    # 绘制衣服区域
                    upper_bbox = clothing_regions['upper']['bbox']
                    lower_bbox = clothing_regions['lower']['bbox']
                    
                    cv2.rectangle(result_img, 
                                (upper_bbox[0], upper_bbox[1]), 
                                (upper_bbox[2], upper_bbox[3]), 
                                color, 2)  # 上衣框
                    cv2.rectangle(result_img, 
                                (lower_bbox[0], lower_bbox[1]), 
                                (lower_bbox[2], lower_bbox[3]), 
                                color, 2)  # 下装框
                    
                    # 绘制连接线
                    center_face_x = (x1 + x2) // 2
                    center_upper_y = (upper_bbox[1] + upper_bbox[3]) // 2
                    center_lower_y = (lower_bbox[1] + lower_bbox[3]) // 2
                    
                    cv2.line(result_img, (center_face_x, y2), 
                            (center_face_x, upper_bbox[1]), color, 1)
                    cv2.line(result_img, (center_face_x, upper_bbox[3]), 
                            (center_face_x, lower_bbox[1]), color, 1)
                    
                    # 添加文本信息
                    face_text = f"Age: {predicted_age:.1f}, {predicted_gender} ({gender_conf:.2f})"
                    upper_text = f"Upper: {upper_color} ({upper_conf:.2f})"
                    lower_text = f"Lower: {lower_color} ({lower_conf:.2f})"
                    
                    # 计算文本位置
                    text_y1 = y1 - 25 if y1 > 25 else y1 + 25
                    text_y2 = text_y1 + 20
                    text_y3 = text_y2 + 20
                    
                    cv2.putText(result_img, face_text, (x1, text_y1), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(result_img, upper_text, (x1, text_y2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(result_img, lower_text, (x1, text_y3), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 收集信息
                    faces_info.append({
                        'age': predicted_age,
                        'gender': predicted_gender,
                        'gender_conf': gender_conf,
                        'upper_color': upper_color,
                        'upper_conf': upper_conf,
                        'lower_color': lower_color,
                        'lower_conf': lower_conf,
                        'face_bbox': (x1, y1, x2, y2),
                        'upper_bbox': upper_bbox,
                        'lower_bbox': lower_bbox
                    })
            
            result = {
                'image': img_path.name,
                'num_faces': len(faces_info),
                'faces_info': faces_info,
                'result_img': result_img,
                'output_path': str(Path('Main/output/images') / f"result_{img_path.name}")
            }
            self.result_queue.put(result)
            
        except Exception as e:
            print(f"处理图片时出错 {img_path}: {str(e)}")
            self.result_queue.put(None)

class AdaptiveProcessor:
    def __init__(self, devices):
        self.devices = devices
        self.processor = ImageProcessor(devices)
        self.batch_size = 32  # 默认批处理大小
        self.num_workers = min(len(devices) * 2, 8)  # 每个GPU分配2个worker，最多8个
        
    def process_images(self, image_paths):
        total_images = len(image_paths)
        
        # 根据图片数量选择处理策略
        if total_images < 20:
            return self._process_simple(image_paths)
        else:
            return self._process_pipeline(image_paths)
    
    def _process_simple(self, image_paths):
        """简单的多线程处理，适合小批量图片"""
        threads = []
        for img_path in image_paths:
            thread = threading.Thread(
                target=self.processor.process_image,
                args=(img_path,)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        return self._collect_results()
    
    def _process_pipeline(self, image_paths):
        """Pipeline处理，适合大批量图片"""
        input_queue = Queue(maxsize=self.batch_size * 2)
        output_queue = Queue()
        
        # 创建工作线程池
        workers = []
        for _ in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_task,
                args=(input_queue, output_queue),
                daemon=True
            )
            workers.append(worker)
            worker.start()
        
        # 生产者线程
        producer = threading.Thread(
            target=self._producer_task,
            args=(input_queue, image_paths),
            daemon=True
        )
        producer.start()
        
        # 等待所有图片处理完成
        producer.join()
        for _ in range(self.num_workers):
            input_queue.put(None)  # 发送结束信号
        for worker in workers:
            worker.join()
        
        return self._collect_results()
    
    def _worker_task(self, input_queue, output_queue):
        """工作线程任务"""
        while True:
            item = input_queue.get()
            if item is None:
                break
            
            try:
                self.processor.process_image(item)
            except Exception as e:
                print(f"处理图片时出错 {item}: {str(e)}")
            finally:
                input_queue.task_done()
    
    def _producer_task(self, input_queue, image_paths):
        """生产者任务"""
        for path in image_paths:
            input_queue.put(path)
    
    def _collect_results(self):
        """收集处理结果"""
        results = []
        while not self.processor.result_queue.empty():
            result = self.processor.result_queue.get()
            if result is not None:
                results.append(result)
        return results

class MediaProcessor:
    def __init__(self, devices):
        self.devices = devices
        self.processor = ImageProcessor(devices)
        self.batch_size = 32
        self.buffer_size = 32  # 增加缓冲区大小
        self.num_workers = min(len(devices) * 2, 8)
        self.frame_queue = Queue(maxsize=self.buffer_size)
        self.result_queue = Queue()
        self.is_processing = False
        
        # 添加批处理缓存
        self.batch_frames = []
        self.batch_indices = []
        
    def process_media(self, source_path: str, output_path: str, media_type: str = 'auto'):
        """处理媒体文件（图片或视频）
        
        Args:
            source_path: 输入文件路径
            output_path: 输出文件路径
            media_type: 'image', 'video' 或 'auto'（自动检测）
        """
        if media_type == 'auto':
            media_type = self._detect_media_type(source_path)
        
        if media_type == 'image':
            return self._process_image(source_path, output_path)
        elif media_type == 'video':
            return self._process_video(source_path, output_path)
        else:
            raise ValueError(f"不支持的媒体类型: {media_type}")
    
    def _detect_media_type(self, path: str) -> str:
        """自动检测媒体类型"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        ext = Path(path).suffix.lower()
        
        if ext in video_extensions:
            return 'video'
        elif ext in image_extensions:
            return 'image'
        else:
            raise ValueError(f"无法识别的文件类型: {ext}")
    
    def _process_image(self, source_path: str, output_path: str):
        """处理单张图片"""
        processor = AdaptiveProcessor(self.devices)
        results = processor.process_images([Path(source_path)])
        
        if results and len(results) > 0:
            result = results[0]
            cv2.imwrite(output_path, result['result_img'])
            result['output_path'] = output_path  # 添加输出路径到结果字典
            return result
        return None
    
    def _process_video(self, source_path: str, output_path: str):
        """处理视频"""
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {source_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # 创建工作线程
        self.is_processing = True
        workers = []
        for _ in range(self.num_workers):
            worker = threading.Thread(
                target=self._video_worker,
                daemon=True
            )
            workers.append(worker)
            worker.start()
        
        try:
            frame_count = 0
            skip_frames = max(1, fps // 30)  # 根据fps动态调整跳帧
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 跳帧处理
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                
                # 将帧放入队列
                self.frame_queue.put((frame_count, frame))
                frame_count += 1
                
                # 处理结果并写入视频
                while not self.result_queue.empty():
                    idx, processed_frame = self.result_queue.get()
                    out.write(processed_frame)
                
                # 示进度
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"\r处理进度: {progress:.1f}%", end='')
            
            # 等待所有帧处理完成
            while frame_count > 0:
                idx, processed_frame = self.result_queue.get()
                out.write(processed_frame)
                frame_count -= 1
                
        finally:
            self.is_processing = False
            cap.release()
            out.release()
            
            for worker in workers:
                worker.join()
    
    def _video_worker(self):
        """视频处理工作线程"""
        batch_frames = []
        batch_indices = []
        max_batch_size = 4  # 批处理大小
        
        while self.is_processing:
            try:
                # 收集批处理帧
                while len(batch_frames) < max_batch_size:
                    try:
                        frame_idx, frame = self.frame_queue.get(timeout=0.1)
                        batch_frames.append(frame)
                        batch_indices.append(frame_idx)
                    except Empty:
                        break
                
                if not batch_frames:
                    continue
                
                # 批量处理帧
                try:
                    # 转换为批处理格式
                    frames_tensor = torch.stack([
                        torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                        for frame in batch_frames
                    ]).cuda()
                    
                    # 批量人脸检测
                    face_results = self.processor.models['face'].predict(frames_tensor)
                    
                    # 处理每一帧
                    for i, (frame, frame_idx) in enumerate(zip(batch_frames, batch_indices)):
                        result_frame = frame.copy()
                        
                        if len(face_results[i].boxes) > 0:
                            for box in face_results[i].boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                # 处理单个人脸...
                                # (这里保持原有的人脸处理逻辑)
                        
                        self.result_queue.put((frame_idx, result_frame))
                    
                except Exception as e:
                    print(f"批处理帧时出错: {str(e)}")
                    # 出错时返回原始帧
                    for frame_idx, frame in zip(batch_indices, batch_frames):
                        self.result_queue.put((frame_idx, frame))
                
                # 清空批处理缓存
                batch_frames.clear()
                batch_indices.clear()
                
            except Exception as e:
                print(f"视频处理工作线程出错: {str(e)}")
                break

router = APIRouter()

# 获取可用设备
devices = get_device()
processor = MediaProcessor(devices)

# 创建临时文件存储目录
TEMP_DIR = Path("temp/uploads")
RESULT_DIR = Path("temp/results")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def save_temp_file(file: UploadFile) -> Path:
    """保存上传的文件到临时目录"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = TEMP_DIR / f"{timestamp}_{file.filename}"
        
        # 同步读取文件内容
        contents = file.file.read()
        
        # 同步写入文件
        with open(temp_file, "wb") as buffer:
            buffer.write(contents)
            
        return temp_file
    except Exception as e:
        logger.error(f"保存临时文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存临时文件失败: {str(e)}")

def cleanup_temp_files():
    """清理临时文件"""
    for file in TEMP_DIR.glob("*"):
        if file.is_file() and (datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)).days >= 1:
            file.unlink()
    
    for file in RESULT_DIR.glob("*"):
        if file.is_file() and (datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)).days >= 1:
            file.unlink()

def process_faces_info(faces_info):
    """处理人脸信息，确保所有值都是JSON可序列化的，并转换颜色为中文"""
    processed_info = []
    for face in faces_info:
        try:
            # 创建一个带有默认值的基础结构
            processed_face = {
                'age': 0.0,
                'gender': "未知",
                'gender_conf': 0.0,
                'upper_color': "未知",
                'upper_conf': 0.0,
                'lower_color': "未知",
                'lower_conf': 0.0,
                'face_bbox': [0, 0, 0, 0],
                'upper_bbox': [0, 0, 0, 0],
                'lower_bbox': [0, 0, 0, 0]
            }
            
            # 安全地获取值
            if isinstance(face.get('age'), dict):
                processed_face['age'] = float(face['age'].get('value', 0))
            elif face.get('age'):
                processed_face['age'] = float(face['age'])
                
            if isinstance(face.get('gender'), dict):
                processed_face['gender'] = str(face['gender'].get('value', '未知'))
                processed_face['gender_conf'] = float(face['gender'].get('confidence', 0))
            else:
                processed_face['gender'] = str(face.get('gender', '未知'))
                processed_face['gender_conf'] = float(face.get('gender_conf', 0))
            
            # 处理服装信息
            if isinstance(face.get('clothing'), dict):
                if isinstance(face['clothing'].get('upper'), dict):
                    processed_face['upper_color'] = translate_color(str(face['clothing']['upper'].get('color', '未知')))
                    processed_face['upper_conf'] = float(face['clothing']['upper'].get('confidence', 0))
                if isinstance(face['clothing'].get('lower'), dict):
                    processed_face['lower_color'] = translate_color(str(face['clothing']['lower'].get('color', '未知')))
                    processed_face['lower_conf'] = float(face['clothing']['lower'].get('confidence', 0))
            
            # 处理边界框信息
            if isinstance(face.get('face'), dict) and face['face'].get('bbox'):
                processed_face['face_bbox'] = [int(x) for x in face['face']['bbox']]
            elif face.get('face_bbox'):
                processed_face['face_bbox'] = [int(x) for x in face['face_bbox']]
            
            processed_info.append(processed_face)
            
        except Exception as e:
            logger.error(f"处理人脸信息时出错: {str(e)}")
            continue
            
    return processed_info

@router.post("/analyze")
def analyze_image(
    file: UploadFile = File(...),
    mode: str = Form(default="normal")
):
    """
    分析上传的图片
    
    Args:
        file: 上传的图片文件
        mode: 分析模式，可选值：normal（普通模式）或 enhanced（增强模式）
    """
    try:
        # 验证文件类型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图片文件")
            
        # 保存临时文件
        temp_path = save_temp_file(file)
        
        try:
            # 使用图像分析器处理图片
            result = image_analyzer.analyze_image(str(temp_path), mode)
            
            # 打印分析结果用于调试
            logger.info(f"分析结果: {json.dumps(result, ensure_ascii=False)}")
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
                
            return {
                "success": True,
                "data": result
            }
            
        except Exception as e:
            logger.error(f"图像分析失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
                
    except Exception as e:
        logger.error(f"请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "devices": devices}
