# test_gender.py
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import random
import torch

def test_gender_classification():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model_path = 'F:/Desktop/train/output/gender_classification/train/weights/best.pt'
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        print("模型加载成功")
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return
    
    # 测试图片目录
    test_dir = Path('F:/Desktop/train/test/images/gender')
    result_dir = Path('F:/Desktop/train/test/result/gender')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n处理测试图片目录: {test_dir}")
    print(f"结果将保存到: {result_dir}\n")
    
    # 性别标签映射
    gender_labels = ['male', 'female']
    
    # 检查测试图片目录是否存在
    if not test_dir.exists():
        print(f"测试图片目录不存在: {test_dir}")
        return
        
    # 获取所有测试图片
    test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    if not test_images:
        print("未找到任何测试图片(.jpg或.png)")
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
            
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
            
            # 预测
            result = model(img)[0]
            probs = result.probs
            pred_gender = gender_labels[probs.top1]
            confidence = probs.top1conf.item()
            
            # 在图片上绘制结果
            result_img = img.copy()
            text = f"Gender: {pred_gender} ({confidence:.2f})"
            cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            # 保存结果图片
            output_path = result_dir / f"result_{img_path.stem}.jpg"
            cv2.imwrite(str(output_path), result_img)
            
            # 收集结果
            results.append({
                'image': img_path.name,
                'predicted_gender': pred_gender,
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
        f.write("性别预测结果汇总\n")
        f.write("=" * 50 + "\n\n")
        for result in results:
            f.write(f"图片: {result['image']}\n")
            f.write(f"预测性别: {result['predicted_gender']}\n")
            f.write(f"置信度: {result['confidence']:.2f}\n")
            f.write(f"结果图片: {result['output_path']}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\n处理完成！")
    print(f"共处理了 {len(results)} 张图片")
    print(f"结果汇总保存在: {summary_path}")

if __name__ == '__main__':
    test_gender_classification()