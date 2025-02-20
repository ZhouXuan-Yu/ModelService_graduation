"""
配置文件
"""

# 图像分析配置
IMAGE_ANALYSIS_CONFIG = {
    # Qwen-VL 配置
    'qwen_vl': {
        'enabled': True,  # 是否启用 Qwen-VL
        'weight': 0.9,    # Qwen-VL 结果权重
        'model': 'qwen-vl-max-latest',  # 模型版本
        'timeout': 30,    # 超时时间（秒）
        'retry_count': 3  # 重试次数
    },
    
    # 本地模型配置
    'local_model': {
        'weight': 0.1,    # 本地模型结果权重
        'min_confidence': 0.2  # 最小置信度
    },
    
    # 缓存配置
    'cache': {
        'enabled': True,  # 是否启用缓存
        'max_size': 100,  # 最大缓存数量
        'ttl': 3600      # 缓存有效期（秒）
    }
}

# API 密钥配置
API_KEYS = {
    'dashscope': {
        'key': None,  # 将从环境变量获取
        'env_var': 'DASHSCOPE_API_KEY'
    }
} 