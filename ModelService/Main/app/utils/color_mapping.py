"""颜色映射工具"""

# 英文到中文的颜色映射
COLOR_EN_TO_CN = {
    'red': '红色',
    'green': '绿色',
    'blue': '蓝色',
    'yellow': '黄色',
    'purple': '紫色',
    'orange': '橙色',
    'pink': '粉色',
    'brown': '棕色',
    'gray': '灰色',
    'grey': '灰色',
    'white': '白色',
    'black': '黑色',
    'navy': '深蓝色',
    'cyan': '青色',
    'magenta': '品红色',
    'silver': '银色',
    'gold': '金色',
    'beige': '米色',
    'khaki': '卡其色',
    'maroon': '栗色',
    'olive': '橄榄色',
    'aqua': '水蓝色',
    'teal': '青绿色',
    'violet': '紫罗兰色',
    'indigo': '靛蓝色',
    'turquoise': '青绿色',
    'coral': '珊瑚色',
    'crimson': '深红色',
    'azure': '天蓝色',
}

# 中文到英文的颜色映射
COLOR_CN_TO_EN = {v: k for k, v in COLOR_EN_TO_CN.items()}

def translate_color(color: str, to_chinese: bool = True) -> str:
    """转换颜色名称
    
    Args:
        color: 颜色名称
        to_chinese: 是否转换为中文，默认True
    
    Returns:
        转换后的颜色名称，如果找不到对应的转换则返回原值
    """
    if not color:
        return color
        
    color = color.lower().strip()
    
    if to_chinese:
        return COLOR_EN_TO_CN.get(color, color)
    else:
        return COLOR_CN_TO_EN.get(color, color)

def get_similar_colors(color: str, to_chinese: bool = True) -> list:
    """获取相似颜色
    
    Args:
        color: 颜色名称
        to_chinese: 返回结果是否为中文，默认True
    
    Returns:
        相似颜色列表
    """
    # 相似颜色组
    similar_groups = [
        {'red', 'crimson', 'maroon'},  # 红色系
        {'green', 'olive', 'teal'},    # 绿色系
        {'blue', 'navy', 'azure'},     # 蓝色系
        {'yellow', 'gold', 'beige'},   # 黄色系
        {'purple', 'violet', 'indigo'}, # 紫色系
        {'gray', 'grey', 'silver'},    # 灰色系
        {'cyan', 'aqua', 'turquoise'}, # 青色系
    ]
    
    # 如果输入是中文，先转换为英文
    if color in COLOR_CN_TO_EN:
        color = COLOR_CN_TO_EN[color]
    
    color = color.lower()
    similar_colors = set()
    
    # 查找包含目标颜色的组
    for group in similar_groups:
        if color in group:
            similar_colors = group
            break
    
    # 转换结果
    if to_chinese:
        return [COLOR_EN_TO_CN.get(c, c) for c in similar_colors]
    return list(similar_colors)

def is_similar_color(color1: str, color2: str) -> bool:
    """判断两个颜色是否相似
    
    Args:
        color1: 第一个颜色名称
        color2: 第二个颜色名称
    
    Returns:
        bool: 是否相似
    """
    # 如果完全相同
    if color1.lower() == color2.lower():
        return True
        
    # 获取两个颜色的相似色组
    color1_similar = set(get_similar_colors(color1, to_chinese=False))
    color2_similar = set(get_similar_colors(color2, to_chinese=False))
    
    # 如果有交集则认为相似
    return bool(color1_similar & color2_similar) 