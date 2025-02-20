from pathlib import Path
from Main.model.z_model_use.use.test_hunhe import MediaProcessor, get_device

def test_single_image(image_path: str):
    """测试单张图片"""
    # 获取设备
    devices = get_device()
    print(f"使用设备: {devices}")
    
    # 创建处理器
    processor = MediaProcessor(devices)
    
    # 设置输出路径
    output_path = str(Path('Main/output/Minture') / f"result_{Path(image_path).name}")
    
    # 处理图片
    print(f"\n处理图片: {image_path}")
    result = processor.process_media(image_path, output_path, media_type='image')
    
    # 打印结果
    if result:
        print("\n分析结果:")
        print(f"检测到的人数: {result['num_faces']}")
        for face_info in result['faces_info']:
            print("\n人物信息:")
            print(f"  年龄: {face_info['age']:.1f}")
            print(f"  性别: {face_info['gender']} (置信度: {face_info['gender_conf']:.2f})")
            print(f"  上衣颜色: {face_info['upper_color']} (置信度: {face_info['upper_conf']:.2f})")
            print(f"  下衣颜色: {face_info['lower_color']} (置信度: {face_info['lower_conf']:.2f})")
        
        print(f"\n结果图片已保存至: {output_path}")
    else:
        print("处理失败")

if __name__ == '__main__':
    # 这里替换为你要测试的图片路径
    test_image = "Main/input/test/images/Minture/test1.png"
    test_single_image(test_image)  