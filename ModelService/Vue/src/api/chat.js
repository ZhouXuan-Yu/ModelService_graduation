import request from '@/utils/request'

export const chatApi = {
  // 使用统一的chat接口
  sendMessage(data) {
    return request.post('/api/chat/completions/', {
      messages: data.messages,
      model: data.model || 'gemma2:2b',
      mode: data.type || 'chat'
    })
  },
  
  // 获取历史记录
  async getHistory(type) {
    try {
      const response = await request.get(`/api/chat/history?type=${type}`)
      return response.data || []
    } catch (error) {
      console.error('获取聊天历史失败:', error)
      return []
    }
  },
  
  // 保存历史记录
  async saveHistory(type, messages) {
    try {
      await request.post('/api/chat/history', {
        type,
        messages
      })
    } catch (error) {
      console.error('保存聊天历史失败:', error)
    }
  }
} 