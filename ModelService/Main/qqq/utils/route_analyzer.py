import re
from typing import Dict, Optional

class RouteAnalyzer:
    def __init__(self):
        self.location_patterns = {
            'start': r'从(.*?)(?:到|去|至|出发)',
            'end': r'(?:到|去|至)(.*?)(?:$|，|。)',
        }

    def extract_locations(self, text: str) -> Dict[str, Optional[str]]:
        locations = {
            'start': None,
            'end': None
        }
        
        # 提取起点
        start_match = re.search(self.location_patterns['start'], text)
        if start_match:
            locations['start'] = start_match.group(1).strip()
            
        # 提取终点
        end_match = re.search(self.location_patterns['end'], text)
        if end_match:
            locations['end'] = end_match.group(1).strip()
            
        return locations

    def analyze_route(self, text: str) -> Dict:
        locations = self.extract_locations(text)
        return {
            'route_info': {
                'start_point': locations['start'],
                'end_point': locations['end'],
                'original_text': text
            }
        }