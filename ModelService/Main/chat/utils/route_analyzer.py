from typing import List, Dict
import json

class RouteAnalyzer:
    def __init__(self):
        self.patterns = {
            'time_sensitive': ['快', '急', '尽快', '最快'],
            'cost_sensitive': ['省钱', '便宜', '经济', '避免收费'],
            'comfort_sensitive': ['舒适', '方便', '简单']
        }
    
    def analyze_chat_history(self, history: List[Dict]) -> Dict:
        """分析聊天历史，返回用户偏好"""
        preferences = {
            'time_sensitive': 0,
            'cost_sensitive': 0,
            'comfort_sensitive': 0,
            'frequent_routes': []
        }
        
        for chat in history:
            if 'text' in chat:
                preferences = self._analyze_text(chat['text'], preferences)
            if 'route_data' in chat:
                preferences = self._analyze_route(chat['route_data'], preferences)
                
        return preferences
    
    def generate_suggestions(self, preferences: Dict) -> List[Dict]:
        """根据用户偏好生成建议"""
        suggestions = []
        
        # 根据偏好生成建议
        if preferences['time_sensitive'] > preferences['cost_sensitive']:
            suggestions.append({
                'type': 'quick_action',
                'title': '快速通勤',
                'description': '为您规划最快捷的通勤路线'
            })
            
        if preferences['cost_sensitive'] > 0:
            suggestions.append({
                'type': 'quick_action',
                'title': '经济路线',
                'description': '为您规划省钱的路线'
            })
            
        if len(preferences['frequent_routes']) > 0:
            suggestions.append({
                'type': 'frequent_route',
                'routes': preferences['frequent_routes'][:3],
                'description': '您的常用路线'
            })
            
        return suggestions 