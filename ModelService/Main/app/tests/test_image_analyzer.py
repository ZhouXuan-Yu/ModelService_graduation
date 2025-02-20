import asyncio
import os
import sys
from pathlib import Path
import torch
import torchvision.models as models
import cv2
import numpy as np
from Main.model.z_model_use.use.test_hunhe import test_comprehensive

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from app.utils.image_analyzer import ImageAnalyzer

# 获取测试图片路径
TEST_DIR = Path(__file__).parent
TEST_IMAGE = str(TEST_DIR / "test1.png")

def print_color_info(color_data):
    """打印颜色信息"""
    if color_data and color_data.get("color"):
        print(f"    颜色: {color_data['color']}")
        print(f"    置信度: {color_data['confidence']:.2f}")
        return True
    return False

async def test_all_models():
    """测试所有模型的综合功能"""
    # 创建分析器实例
    analyzer = ImageAnalyzer()
    
    # 确保测试图片存在
    if not os.path.exists(TEST_IMAGE):
        print(f"警告: 测试图片不存在: {TEST_IMAGE}")
        return
    
    try:
        # 1. 测试模型加载
        print("\n=== 测试模型加载 ===")
        for model_name, model_path in analyzer.model_paths.items():
            if model_path and os.path.exists(model_path):
                print(f"{model_name} 模型文件存在: {model_path}")
            else:
                print(f"警告: {model_name} 模型文件不存在: {model_path}")
        
        # 2. 测试图像分析
        print("\n=== 测试图像分析 ===")
        result = await analyzer.analyze_image(TEST_IMAGE)
        
        # 验证返回结果的结构
        assert isinstance(result, dict)
        assert "num_persons" in result
        assert "persons" in result
        assert "segmentation" in result
        
        # 打印分析结果
        print("\n分析摘要:")
        print(result["summary"])
        
        # 打印详细的人物信息
        print("\n详细信息:")
        for person in result["persons"]:
            print(f"\n人物 {person['person_id']}:")
            print(f"  人脸位置: {person['face']['bbox']}")
            print(f"  人脸检测置信度: {person['face']['confidence']:.2f}")
            
            if person['gender']['detected']:
                print(f"  性别: {person['gender']['value']} (置信度: {person['gender']['confidence']:.2f})")
            else:
                print("  性别: 未检测到")
            
            if person['age']['detected']:
                print(f"  年龄: {person['age']['value']:.1f}岁 (置信度: {person['age']['confidence']:.2f})")
            else:
                print("  年龄: 未检测到")
            
            print("  衣着:")
            if person['clothing']['upper']['detected']:
                print(f"    上衣: {person['clothing']['upper']['color']} (置信度: {person['clothing']['upper']['confidence']:.2f})")
            else:
                print("    上衣: 未检测到")
            
            if person['clothing']['lower']['detected']:
                print(f"    下衣: {person['clothing']['lower']['color']} (置信度: {person['clothing']['lower']['confidence']:.2f})")
            else:
                print("    下衣: 未检测到")
        
        return True
    
    except Exception as e:
        print(f"测试失败: {str(e)}")
        return False

def test_analyzer():
    """测试图像分析器"""
    print("\n=== 测试模型加载 ===")
    
    # 检查模型文件是否存在
    model_files = {
        'face': 'Main/model/output/face_detection/train2/weights/best.pt',
        'color': 'Main/model/output/color_classification/best_model.pth',
        'age': 'Main/model/output/age_estimation/weights/best.pt',
        'gender': 'Main/model/output/gender_classification/train/weights/best.pt'
    }
    
    for name, path in model_files.items():
        if Path(path).exists():
            print(f"{name} 模型文件存在: {path}")
        else:
            print(f"{name} 模型文件不存在: {path}")
    
    print("\n=== 测试图像分析 ===")
    test_comprehensive()

if __name__ == '__main__':
    test_analyzer() 