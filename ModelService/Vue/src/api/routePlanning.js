import request from './request'

export const routePlanningApi = {
  // 获取路线规划
  async getRoutePlan(data) {
    try {
      console.log('发送路线规划请求:', data)  // 添加调试日志
      
      const response = await request.post('/api/route/plan', {
        text: data.text,
        model: 'gemma2:2b'
      }, {
        timeout: 400000  // 设置为400秒
      })
      
      console.log('路线规划响应:', response)  // 添加调试日志
      
      if (!response.success) {
        throw new Error(response.error || '路线规划失败')
      }
      return response
    } catch (error) {
      console.error('路线规划失败:', error)
      throw error
    }
  }
} 