import json

# 读取城市数据
with open('cities.json', 'r', encoding='utf-8') as f:
    cities = json.load(f)

# 使用示例
def get_city_name(code):
    """根据城市代码获取城市名称"""
    return cities.get(str(code), {}).get('name')

def get_city_pinyin(code):
    """根据城市代码获取城市拼音"""
    return cities.get(str(code), {}).get('city')

# 示例
print(get_city_name('1100'))  # 输出: 北京市
print(get_city_pinyin('1100'))  # 输出: beijing