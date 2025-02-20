ROUTE_SUGGESTIONS = {
    'patterns': [
        {
            'keywords': ['上班', '通勤'],
            'suggestions': [
                '需要考虑早高峰路况吗？',
                '是否需要避开拥堵路段？',
                '要优先选择最快还是最省钱的路线？'
            ]
        },
        {
            'keywords': ['旅游', '游玩', '景点'],
            'suggestions': [
                '是否需要规划停车位？',
                '是否需要避开景点高峰期？',
                '要按照特定顺序游览吗？'
            ]
        }
    ],
    'default_suggestions': [
        '您可以告诉我起点和终点',
        '可以指定途经点',
        '可以选择是否避开收费路段'
    ]
}

QUICK_ACTIONS = [
    {
        'id': 'commute',
        'title': '通勤路线',
        'description': '智能规划上下班路线',
        'prompt': '帮我规划一条从 {start} 到 {end} 的通勤路线，考虑实时路况'
    },
    {
        'id': 'economic',
        'title': '经济路线',
        'description': '优先选择免费道路',
        'prompt': '帮我规划一条从 {start} 到 {end} 的经济路线，尽量避免收费'
    },
    {
        'id': 'tourism',
        'title': '景点游览',
        'description': '合理规划游览顺序',
        'prompt': '帮我规划一条游览路线，起点是 {start}，需要游览这些景点: {spots}'
    }
] 