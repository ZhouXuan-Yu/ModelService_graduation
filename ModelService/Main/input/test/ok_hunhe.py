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
import time

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
            print(f"��� {num_gpus} 个 GPU:")
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
        self.load_model()
    
    def load_model(self):
        self.model = YOLO(self.model_path)
    
    def predict(self, img):
        return self.model(img)

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
                
                # 修改最后的全连接层，匹配训练的���
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
        self.confidence_threshold = 0.7  # 提高基础阈值
        self.load_model()
        # 定义增强的预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
    
    def preprocess_face(self, img):
        """增强的人脸预处理"""
        # 转换为PIL图像
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 应用预处理转换
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor
    
    def multi_scale_predict(self, img):
        """多尺度预测"""
        # 定义不同的尺度和旋转角度
        scales = [0.9, 1.0, 1.1]  # 缩小尺度范围，更关注接近原始尺寸的预测
        rotations = [-5, 0, 5]    # 减小旋转角度，更关注接近正面的预测
        predictions = []
        confidences = []
        
        # 原始图像预测
        for scale in scales:
            # 调整图像大小
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = cv2.resize(img, (new_w, new_h))
            
            # 对每个旋转角度进行预测
            for angle in rotations:
                if angle != 0:
                    # 计算旋转矩阵
                    center = (new_w // 2, new_h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_img = cv2.warpAffine(scaled_img, M, (new_w, new_h))
                else:
                    rotated_img = scaled_img
                
                # 转换为PIL图像并预处理
                pil_img = Image.fromarray(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
                
                # 预测
                results = self.model(pil_img)
                if len(results) > 0:
                    probs = results[0].probs
                    if probs is not None:
                        gender_idx = int(probs.top1)
                        confidence = float(probs.top1conf)
                        
                        # 据尺度和角度调整置信度
                        if scale != 1.0:
                            confidence *= 0.95  # 提高非原始尺度的权重
                        if angle != 0:
                            confidence *= 0.95  # 提高旋转角度的权重
                        
                        predictions.append(gender_idx)
                        confidences.append(confidence)
        
        if not predictions:
            return "未知", 0.0
        
        # 加权投票系统
        weighted_votes = np.zeros(2)  # 两个类别
        total_weight = sum(confidences)
        
        if total_weight == 0:
            return "未知", 0.0
        
        # 计算加权投票
        for pred, conf in zip(predictions, confidences):
            weighted_votes[pred] += conf
        
        # 获取最终预测
        final_pred = np.argmax(weighted_votes)
        final_conf = weighted_votes[final_pred] / total_weight
        
        # 应用动态置信度阈值
        # 根据预测的一致性调整阈值，范围更窄以保持稳定性
        vote_ratio = max(weighted_votes) / sum(weighted_votes)
        dynamic_threshold = self.confidence_threshold * (0.9 + 0.2 * vote_ratio)  # 范围在0.9-1.1之间
        
        if final_conf < dynamic_threshold:
            return "未知", final_conf
        
        # 额外的一致性检查
        if vote_ratio < 0.75:  # 如果最高票数不足75%，认为预测不够可靠
            return "未知", final_conf
        
        return "男" if final_pred == 0 else "女", final_conf
    
    def predict(self, img):
        """使用多尺度和强预测进行性别分类"""
        return self.multi_scale_predict(img)

class ClothingDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # 加载语义分割模型（可选）
        self.segmentation_model = None
        try:
            weights_path = Path('F:/Desktop/train/models/deeplabv3_resnet50_coco-cd0a2569.pth')
            if not weights_path.exists():
                print("提示: 未找到DeepLabV3模型，将使用简单的区域检测方法")
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
                self.segmentation_model.eval()
                print("成功加载分割模型")
        except Exception as e:
            print(f"加载分割模型出错: {str(e)}")
            print("将使用简单的区域检测方法")
            self.segmentation_model = None
    
    def detect_clothing_regions(self, img, face_box):
        """检测上衣和下装区域"""
        x1, y1, x2, y2 = face_box
        face_height = y2 - y1
        face_width = x2 - x1
        h, w = img.shape[:2]
        
        # 使用简单的几何方法估计衣服区域
        # 上衣区域：从脸部底开始，向下延伸1.5倍人脸高度
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
                # 转换为RGB格式
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                
                with torch.no_grad():
                    output = self.segmentation_model(img_tensor)['out'][0]
                    segmentation = torch.argmax(output, dim=0).cpu().numpy()
                
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
        model_configs = {
            'face': {
                'path': 'F:/Desktop/train/output/face_detection/train2/weights/best.pt',
                'class': FaceModel
            },
            'color': {
                'path': 'F:/Desktop/train/output/color_classification/best_model.pth',
                'class': ColorModel
            },
            'age': {
                'path': 'F:/Desktop/train/output/age_estimation/weights/best.pt',
                'class': AgeModel
            },
            'gender': {
                'path': 'F:/Desktop/train/output/gender_classification/train/weights/best.pt',
                'class': GenderModel
            }
        }
        
        for model_name, config in model_configs.items():
            try:
                device = self.get_next_device()
                print(f"正在加载 {model_name} 模型到设备 {device}...")
                self.models[model_name] = config['class'](config['path'], device)
                print(f"{model_name} 模型加载成功")
            except Exception as e:
                print(f"{model_name} 模型加载失败: {str(e)}")
                raise
    
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
                        lower_color, lower_conf = "未", 0.0
                    
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
                'output_path': str(Path('F:/Desktop/train/test/result/Minture') / f"result_{img_path.name}")
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
        
        # 添加缓存和预测相关的属性
        self.frame_cache = {}  # 用于存储处理结果的缓存
        self.cache_size = 100  # 缓存大小限制
        self.prediction_window = 5  # 预测窗口大小
        self.last_results = []  # 存储最近的处理结果
        self.motion_threshold = 0.1  # 运动检测阈值
        
    def _calculate_frame_difference(self, frame1, frame2):
        """计算两帧之间的差异"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff) / 255.0
    
    def _predict_next_frame(self, current_results, frame):
        """预测下一帧的结果"""
        if not current_results or not self.last_results:
            return None
            
        # 计算人脸位置的移动趋势
        movements = []
        for prev, curr in zip(self.last_results[-self.prediction_window:], current_results):
            if prev and curr:
                dx = curr['x'] - prev['x']
                dy = curr['y'] - prev['y']
                movements.append((dx, dy))
        
        if not movements:
            return None
            
        # 计算平均移动
        avg_dx = sum(m[0] for m in movements) / len(movements)
        avg_dy = sum(m[1] for m in movements) / len(movements)
        
        # 预测下一帧的位置
        predicted_results = []
        for curr in current_results:
            pred = curr.copy()
            pred['x'] += avg_dx
            pred['y'] += avg_dy
            predicted_results.append(pred)
            
        return predicted_results
    
    def _update_cache(self, frame_idx, result):
        """更新缓存"""
        self.frame_cache[frame_idx] = result
        if len(self.frame_cache) > self.cache_size:
            oldest_key = min(self.frame_cache.keys())
            del self.frame_cache[oldest_key]
    
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
        """处理视频（优化版本）"""
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
            last_frame = None
            last_processed_frame = None
            skip_frames = max(1, fps // 30)  # 动态跳帧
            timeout = 30.0  # 设置超时时间（秒）
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检查缓存
                if frame_count in self.frame_cache:
                    out.write(self.frame_cache[frame_count])
                    frame_count += 1
                    continue
                
                # 运动检测
                if last_frame is not None:
                    diff = self._calculate_frame_difference(last_frame, frame)
                    if diff < self.motion_threshold and last_processed_frame is not None:
                        out.write(last_processed_frame)
                        self._update_cache(frame_count, last_processed_frame)
                        frame_count += 1
                        continue
                
                # 跳帧处理
                if frame_count % skip_frames != 0:
                    if last_processed_frame is not None:
                        out.write(last_processed_frame)
                        self._update_cache(frame_count, last_processed_frame)
                    frame_count += 1
                    continue
                
                try:
                    # 将帧放入队列，设置超时
                    self.frame_queue.put((frame_count, frame), timeout=1.0)
                    
                    # 等待结果，设置超时
                    start_time = time.time()
                    while True:
                        try:
                            idx, result_frame = self.result_queue.get(timeout=1.0)
                            if idx == frame_count:
                                break
                            if time.time() - start_time > timeout:
                                print(f"处理帧 {frame_count} 超时，使用原始帧")
                                result_frame = frame
                                break
                        except Empty:
                            if time.time() - start_time > timeout:
                                print(f"处理帧 {frame_count} 超时，使用原始帧")
                                result_frame = frame
                                break
                            continue
                    
                    # 更新缓存和状态
                    self._update_cache(frame_count, result_frame)
                    last_processed_frame = result_frame
                    last_frame = frame
                    
                    out.write(result_frame)
                    frame_count += 1
                    
                    # 显示进度
                    if frame_count % 30 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"\r处理进度: {progress:.1f}%", end='')
                    
                except Exception as e:
                    print(f"处理帧 {frame_count} 时出错: {str(e)}")
                    if last_processed_frame is not None:
                        out.write(last_processed_frame)
                    else:
                        out.write(frame)
                    frame_count += 1
                    continue
            
        finally:
            self.is_processing = False
            cap.release()
            out.release()
            
            # 等待所有工作线程结束，设置超时
            for worker in workers:
                worker.join(timeout=5.0)
    
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
                        if batch_frames:
                            break
                        continue
                
                if not batch_frames:
                    continue
                
                # 批量处理帧
                try:
                    # 预处理每一帧到正确的尺寸
                    processed_frames = []
                    for frame in batch_frames:
                        # 调整图像大小为640x640（YOLO的标准输入尺寸）
                        h, w = frame.shape[:2]
                        # 保持宽高比
                        ratio = min(640/h, 640/w)
                        new_h, new_w = int(h * ratio), int(w * ratio)
                        # ��整大小
                        resized = cv2.resize(frame, (new_w, new_h))
                        # 创建640x640的画布
                        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
                        # 将调整后的图像放在画布中央
                        y_offset = (640 - new_h) // 2
                        x_offset = (640 - new_w) // 2
                        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                        processed_frames.append(canvas)
                    
                    # 转换为批处理格式
                    frames_tensor = torch.stack([
                        torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                        for frame in processed_frames
                    ]).cuda()
                    
                    # 批量人脸检测
                    face_results = self.processor.models['face'].predict(frames_tensor)
                    
                    # 处理每一帧
                    for i, (frame, frame_idx) in enumerate(zip(batch_frames, batch_indices)):
                        result_frame = frame.copy()
                        orig_h, orig_w = frame.shape[:2]
                        
                        if len(face_results[i].boxes) > 0:
                            # 计算缩放比例
                            scale_x = orig_w / 640
                            scale_y = orig_h / 640
                            
                            for box in face_results[i].boxes:
                                # 将预测坐标转换回原始图像尺寸
                                x1, y1, x2, y2 = map(float, box.xyxy[0])
                                x1 = int((x1 - x_offset) * scale_x)
                                x2 = int((x2 - x_offset) * scale_x)
                                y1 = int((y1 - y_offset) * scale_y)
                                y2 = int((y2 - y_offset) * scale_y)
                                
                                # 确保坐标在有效范围内
                                x1 = max(0, min(x1, orig_w))
                                x2 = max(0, min(x2, orig_w))
                                y1 = max(0, min(y1, orig_h))
                                y2 = max(0, min(y2, orig_h))
                                
                                # 处理单个人脸
                                face_img = frame[y1:y2, x1:x2]
                                if face_img.size == 0:
                                    continue
                                
                                # 转换为RGB格式
                                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                face_pil = Image.fromarray(face_rgb)
                                
                                # 预测年龄和性别
                                predicted_age = self.processor.models['age'].predict(face_pil)
                                predicted_gender, gender_conf = self.processor.models['gender'].predict(face_img)
                                
                                # 绘制结果
                                color = (0, 255, 0) if predicted_gender != "未知" else (0, 165, 255)
                                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                                
                                # 添加文本信息
                                text = f"Age: {predicted_age:.1f}, {predicted_gender} ({gender_conf:.2f})"
                                cv2.putText(result_frame, text, (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
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
                if not self.is_processing:
                    break
                continue
    
    def _apply_predictions(self, frame, predictions):
        """应用预测结果到当前帧"""
        result_frame = frame.copy()
        
        for pred in predictions:
            # 应用预测的人脸位置和其他属性
            x1, y1 = int(pred['x']), int(pred['y'])
            x2, y2 = x1 + pred['width'], y1 + pred['height']
            
            # 绘制预测的结果
            color = (0, 255, 0)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # 添加预测的文本信息
            if 'text' in pred:
                cv2.putText(result_frame, pred['text'], (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result_frame

def test_comprehensive():
    # 获取可用设备
    devices = get_device()
    
    # 创建媒体处理器
    processor = MediaProcessor(devices)
    
    # 设置路径
    test_dir = Path('F:/Desktop/train/test/images/Minture')
    result_dir = Path('F:/Desktop/train/test/result/Minture')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理所有媒体文件
    media_files = []
    media_files.extend(test_dir.glob('*.jpg'))
    media_files.extend(test_dir.glob('*.png'))
    media_files.extend(test_dir.glob('*.mp4'))
    media_files.extend(test_dir.glob('*.avi'))
    
    if not media_files:
        print("未找到任何媒体文件")
        return
    
    print(f"\n开始处理 {len(media_files)} 个文件...")
    
    # 处理每个文件
    results = []
    for media_path in media_files:
        try:
            print(f"\n处理文件: {media_path.name}")
            output_path = str(result_dir / f"result_{media_path.name}")
            result = processor.process_media(str(media_path), output_path)
            if result:
                result['output_path'] = output_path  # 确保结果中包含输出路径
                results.append(result)
        except Exception as e:
            print(f"处理文件时出错 {media_path}: {str(e)}")
    
    # 保存汇总结果
    summary_path = result_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("综合预测结果汇总\n")
        f.write("=" * 50 + "\n\n")
        for result in results:
            if isinstance(result, dict):  # 图片结果
                f.write(f"图片: {result['image']}\n")
                f.write(f"检测到的人脸数量: {result['num_faces']}\n")
                if result['faces_info']:
                    f.write("人脸信息:\n")
                    for i, face in enumerate(result['faces_info'], 1):
                        f.write(f"  人脸 {i}:\n")
                        f.write(f"    年龄: {face['age']:.1f}\n")
                        f.write(f"    性别: {face['gender']} (置信度: {face['gender_conf']:.2f})\n")
                        f.write(f"    上装颜色: {face['upper_color']} (置信度: {face['upper_conf']:.2f})\n")
                        f.write(f"    下装颜色: {face['lower_color']} (置信度: {face['lower_conf']:.2f})\n")
                f.write(f"结果文件: {result['output_path']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\n处理完成！")
    print(f"共处理了 {len(results)} 个文件")
    print(f"结果汇总保存在: {summary_path}")

if __name__ == '__main__':
    test_comprehensive()
