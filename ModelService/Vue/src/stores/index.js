import { defineStore } from 'pinia'
import request from '@/utils/request'
import { routePlanningApi } from '@/api/routePlanning'

export const useMainStore = defineStore('main', {
  state: () => ({
    chatHistory: [],
    currentChatType: null,
    loading: false,
    error: null
  }),
  
  actions: {
    async loadChatHistory(type) {
      this.loading = true
      try {
        const response = await request.get('/chat/completions/history', {
          params: { 
            type: type || 'general'
          }
        })
        
        this.chatHistory = response?.history || []
        this.currentChatType = type
      } catch (error) {
        console.error('加载聊天历史失败:', error)
        this.error = error.message
        this.chatHistory = []
      } finally {
        this.loading = false
      }
    },

    async sendChatMessage({ type, content }) {
      try {
        let response
        if (type === 'route') {
          // 路径规划请求
          try {
            response = await routePlanningApi.getRoutePlan({
              text: content
            })
            
            if (response.success && response.route_data) {
              this.chatHistory.push({
                role: 'user',
                content
              }, {
                role: 'assistant',
                content: `已为您规划从 ${response.route_data.route_info.start_point} 到 ${response.route_data.route_info.end_point} 的路线`,
                route_data: response.route_data
              })
            } else {
              // 添加错误消息到聊天历史
              this.chatHistory.push({
                role: 'user',
                content
              }, {
                role: 'assistant',
                content: `抱歉，路线规划失败：${response.error || '未知错误'}`
              })
            }
          } catch (error) {
            // 添加错误消息到聊天历史
            this.chatHistory.push({
              role: 'user',
              content
            }, {
              role: 'assistant',
              content: `抱歉，发生错误：${error.message}`
            })
            throw error
          }
        } else {
          // 普通聊天请求
          response = await request.post('/chat/completions', {
            messages: [{
              role: 'user',
              content: content
            }],
            model: 'gemma2:2b',
            type: type
          })
          
          if (response?.message) {
            this.chatHistory.push({
              role: 'user',
              content
            }, {
              role: 'assistant',
              content: response.message
            })
          }
        }
        
        return response
      } catch (error) {
        console.error('发送消息失败:', error)
        throw error
      }
    }
  }
}) 