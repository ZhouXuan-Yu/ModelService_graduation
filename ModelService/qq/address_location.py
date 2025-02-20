import requests
import json
from typing import Tuple, Optional

class AddressLocationUtil:
    """
    地址转经纬度（使用高德地图获取地址信息的经纬度）
    """
    
    # 高德地图API密钥
    KEY = "206278d547a0c6408987f2a0002e2243"
    
    # API请求URL模板
    GD_URL = "https://restapi.amap.com/v3/geocode/geo"
    
    # 成功标识
    SUCCESS_FLAG = "1"
    
    @staticmethod
    def get_lon_and_lat_by_address(address: str) -> Optional[Tuple[float, float]]:
        """
        根据地址获取对应的经纬度信息
        
        Args:
            address: 地址字符串
            
        Returns:
            Tuple[float, float]: 经度和纬度的元组，如果转换失败返回None
        """
        try:
            # 构建请求参数
            params = {
                "address": address,
                "key": AddressLocationUtil.KEY
            }
            
            # 发送GET请求到高德API
            response = requests.get(AddressLocationUtil.GD_URL, params=params)
            response.raise_for_status()
            
            # 解析JSON响应
            result = response.json()
            
            if str(result.get("status")) == AddressLocationUtil.SUCCESS_FLAG:
                if result.get("geocodes") and len(result["geocodes"]) > 0:
                    location = result["geocodes"][0].get("location", "")
                    if location:
                        lon, lat = map(float, location.split(","))
                        return lon, lat
            return None
                
        except Exception as e:
            print(f"转换地址时发生错误: {str(e)}")
            return None

def create_track_data(start_address: str, end_address: str) -> dict:
    """
    创建轨迹数据
    
    Args:
        start_address: 起点地址
        end_address: 终点地址
        
    Returns:
        dict: 包含轨迹数据的字典
    """
    # 获取起点经纬度
    start_coords = AddressLocationUtil.get_lon_and_lat_by_address(start_address)
    if not start_coords:
        raise ValueError(f"无法获取起点 '{start_address}' 的经纬度")
    
    # 获取终点经纬度
    end_coords = AddressLocationUtil.get_lon_and_lat_by_address(end_address)
    if not end_coords:
        raise ValueError(f"无法获取终点 '{end_address}' 的经纬度")
    
    # 创建轨迹数据
    track_data = {
        "name": f"{start_address} -> {end_address}",
        "path": [
            list(start_coords),  # 起点坐标
            list(end_coords)     # 终点坐标
        ]
    }
    
    return track_data

def main():
    """
    主函数：从终端输入地址并生成轨迹数据
    """
    try:
        # 获取用户输入
        start_address = input("请输入起点地址: ").strip()
        end_address = input("请输入终点地址: ").strip()
        
        # 创建轨迹数据
        track_data = create_track_data(start_address, end_address)
        
        # 打印结果
        print("\n轨迹数据:")
        print(json.dumps(track_data, ensure_ascii=False, indent=2))
        
        # 打印详细信息
        print("\n详细信息:")
        print(f"起点: {start_address}")
        print(f"起点坐标: {track_data['path'][0]}")
        print(f"终点: {end_address}")
        print(f"终点坐标: {track_data['path'][1]}")
        
        return track_data
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return None

if __name__ == "__main__":
    main() 