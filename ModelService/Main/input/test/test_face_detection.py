import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='人脸检测测试')
    parser.add_argument('--weights', type=str, default='F:/Desktop/train/output/face_detection/train2/weights/best.pt',
                      help='模型权重路径')
    parser.add_argument('--source', type=str, default='F:/Desktop/train/test/images/face',
                      help='测试图像路径/文件夹/视频')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='置信度阈值')
    parser.add_argument('--device', type=str, default='auto',
                      help='测试设备 (auto/cpu/0,1,2...)')
    parser.add_argument('--save-txt', action='store_true',
                      help='保存预测结果为txt文件')
    parser.add_argument('--save-conf', action='store_true',
                      help='在txt文件中保存置信度')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 确保权重文件存在
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"错误：模型权重不存在: {weights_path}")
        return
    
    # 加载模型
    model = YOLO(str(weights_path))
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 创建输出目录
    output_dir = Path('F:/Desktop/train/test/result/face')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行预测
    results = model.predict(
        source=args.source,
        conf=args.conf,
        device=device,
        save=True,  # 保存预测结果
        save_txt=args.save_txt,  # 保存txt结果
        save_conf=args.save_conf,  # 保存置信度
        project=str(output_dir),
        name='predict',
        exist_ok=True
    )
    
    # 打印预测结果统计
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\n预测完成!")
    print(f"总检测数: {total_detections}")
    
    # 计算平均置信度
    if total_detections > 0:
        avg_conf = sum(box.conf.item() for r in results for box in r.boxes) / total_detections
        print(f"平均置信度: {avg_conf:.3f}")

if __name__ == '__main__':
    main() 