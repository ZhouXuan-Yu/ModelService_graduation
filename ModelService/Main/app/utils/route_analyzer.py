import re
from typing import Dict, Optional, List

class RouteAnalyzer:
    def __init__(self):
        self.patterns = {
            'return': r'返程|回去|原路返回',
            'same_route': r'同样|相同|一样的路线',
            'time_priority': r'最快|尽快|赶时间',
            'cost_priority': r'最便宜|省钱|经济',
            'distance_priority': r'最近|近路|短路',
            'avoid_highway': r'不要高速|避开高速|不走高速',
            'avoid_toll': r'不要收费|免费|不收费',
            'avoid_traffic': r'避开拥堵|避开堵车|不要堵'
        }
    
    def analyze_route_text(self, text: str, historical_route: Optional[Dict] = None) -> Dict:
        """分析路线文本"""
        result = {
            'is_return': bool(re.search(self.patterns['return'], text)),
            'use_same_route': bool(re.search(self.patterns['same_route'], text)),
            'preferences': {
                'time_priority': bool(re.search(self.patterns['time_priority'], text)),
                'cost_priority': bool(re.search(self.patterns['cost_priority'], text)),
                'distance_priority': bool(re.search(self.patterns['distance_priority'], text)),
                'avoid_highway': bool(re.search(self.patterns['avoid_highway'], text)),
                'avoid_toll': bool(re.search(self.patterns['avoid_toll'], text)),
                'avoid_traffic': bool(re.search(self.patterns['avoid_traffic'], text))
            }
        }
        
        # 处理返程和相同路线的情况
        if historical_route:
            if result['is_return']:
                result['start_point'] = historical_route['end_point']
                result['end_point'] = historical_route['start_point']
            elif result['use_same_route']:
                result['start_point'] = historical_route['start_point']
                result['end_point'] = historical_route['end_point']
        
        return result 