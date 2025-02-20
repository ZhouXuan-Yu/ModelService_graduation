# test_color.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import os
from pathlib import Path
import random
import timm

class ColorClassifier(nn.Module):
    """颜色分类模型"""
    def __init__(self, num_classes):
        super(ColorClassifier, self).__init__()
        # 使用ResNet18作为基础模型
        self.model = models.resnet18(pretrained=False)
        
        # 修改最后的全连接层，匹配训练时的结构
        # 使用与保存的模型完全相同的层索引
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential()
        self.model.fc.add_module('1', nn.Linear(num_features, 512))
        self.model.fc.add_module('2', nn.ReLU())
        self.model.fc.add_module('4', nn.Linear(512, num_classes))
    
    def forward(self, x):
        return self.model(x)

def test_color_classification():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model_path = 'F:/Desktop/train/output/color_classification/best_model.pth'
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        print(f"Loading model from: {model_path}")
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)
        
        # 从检查点获取类别
        if 'classes' in checkpoint:
            classes = checkpoint['classes']
            print(f"\n加载到的颜色类别 ({len(classes)} 类):")
            for i, cls in enumerate(classes):
                print(f"{i}: {cls}")
        else:
            raise ValueError("检查点中没有找到类别信息")
        
        # 创建模型
        model = ColorClassifier(num_classes=len(classes))
        
        # 打印模型的状态字典键以行调试
        print("\n当前模型的键:")
        current_keys = set(model.state_dict().keys())
        for key in current_keys:
            print(f"- {key}")
            
        print("\n检查点中的键:")
        checkpoint_keys = set(checkpoint['model_state_dict'].keys())
        for key in checkpoint_keys:
            print(f"- {key}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("\n模型加载成功")
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return
    
    # 设置图像转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 测试图片目录
    test_dir = Path('F:/Desktop/train/test/images/color')
    result_dir = Path('F:/Desktop/train/test/result/color')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n处理测试图片目录: {test_dir}")
    print(f"结果将保存到: {result_dir}\n")
    
    # 检查测试图片目录是否存在
    if not test_dir.exists():
        print(f"测试图片目录不存在: {test_dir}")
        return
        
    # 获取所有测试图片
    test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    if not test_images:
        print("未到任何测试图片(.jpg或.png)")
        return
    
    # 随机选择10张图片
    if len(test_images) > 10:
        test_images = random.sample(test_images, 10)
    
    print(f"将处理 {len(test_images)} 张测试图片")
    
    # 处理每张测试图片
    results = []
    for img_path in test_images:
        try:
            print(f"\n处理图片: {img_path.name}")
            
            # 读取并转换图片
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # 读取原始图片用于显示
            img_display = cv2.imread(str(img_path))
            if img_display is None:
                print(f"无法读取图片: {img_path}")
                continue
            
            # 预测
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
                pred_color = classes[pred_idx.item()]
                confidence = confidence.item()
            
            # 在图片上绘制结果
            result_img = img_display.copy()
            text = f"Color: {pred_color} ({confidence:.2f})"
            cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            # 保存结果图片
            output_path = result_dir / f"result_{img_path.stem}.jpg"
            cv2.imwrite(str(output_path), result_img)
            
            # 收集结果
            results.append({
                'image': img_path.name,
                'predicted_color': pred_color,
                'confidence': confidence,
                'output_path': output_path
            })
            
            print(f"预测结果: {text}")
            print(f"结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"处理图片时出错 {img_path}: {str(e)}")
            continue
    
    # 保存汇总结果
    summary_path = result_dir / 'summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("颜色预测结果汇总\n")
        f.write("=" * 50 + "\n\n")
        for result in results:
            f.write(f"图片: {result['image']}\n")
            f.write(f"预测颜色: {result['predicted_color']}\n")
            f.write(f"置信度: {result['confidence']:.2f}\n")
            f.write(f"结果图片: {result['output_path']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\n处理完！")
    print(f"共处理了 {len(results)} 张图片")
    print(f"结果汇总保存在: {summary_path}")

if __name__ == '__main__':
    test_color_classification()