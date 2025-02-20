import os
import shutil
from pathlib import Path
import time

def clean_npy_cache(face_dir: str):
    """
    清理指定目录下的 .npy 缓存文件
    
    Args:
        face_dir: face 目录的路径
    """
    try:
        # 转换为 Path 对象
        face_path = Path(face_dir)
        if not face_path.exists():
            print(f"错误: 目录不存在 - {face_dir}")
            return
        
        # 统计信息
        total_size = 0
        file_count = 0
        start_time = time.time()
        
        # 遍历目录
        print(f"\n开始清理缓存文件...")
        print(f"目标目录: {face_dir}")
        
        for npy_file in face_path.rglob("*.npy"):
            try:
                # 获取文件大小
                file_size = npy_file.stat().st_size
                total_size += file_size
                file_count += 1
                
                # 删除文件
                npy_file.unlink()
                print(f"已删除: {npy_file.relative_to(face_path)} ({file_size / 1024 / 1024:.2f} MB)")
                
            except Exception as e:
                print(f"删除文件时出错 {npy_file}: {str(e)}")
        
        # 计算总耗时
        elapsed_time = time.time() - start_time
        
        # 打印统计信息
        print(f"\n清理完成!")
        print(f"- 总共删除文件数: {file_count}")
        print(f"- 总释放空间: {total_size / 1024 / 1024:.2f} MB")
        print(f"- 耗时: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        print(f"清理过程中出错: {str(e)}")

if __name__ == "__main__":
    # face 目录路径
    FACE_DIR = r"F:\Desktop\train\data\face"
    
    # 执行清理
    clean_npy_cache(FACE_DIR)