import { defineStore } from 'pinia'
import request from '@/utils/request'
import { routePlanningApi } from '@/api/routePlanning'

export const useMainStore = defineStore('main', {
    state: () => ({
        chatHistory: [],
        currentChatType: null,
        loading: false,
        error: null,
        routeSessionStartTime: null, // 当前路线规划会话的开始时间
        selectedRouteRecord: null, // 从路线记录中选择的记录
        routeRecords: [], // 路线记录列表
    }),

    actions: {
        // 开始新的路线规划会话
        startNewRouteSession() {
            this.chatHistory = []
            this.routeSessionStartTime = Date.now()
            this.selectedRouteRecord = null
            return this.routeSessionStartTime
        },

        // 保存当前路线规划记录
        saveRouteRecord(routeInfo) {
            try {
                if (!this.chatHistory.length || !routeInfo) return false

                const currentDate = new Date()
                const dateString = `${currentDate.getFullYear()}-${String(currentDate.getMonth() + 1).padStart(2, '0')}-${String(currentDate.getDate()).padStart(2, '0')}`

                // 获取现有记录
                let records = []
                const storedRecords = localStorage.getItem('routeRecords')
                if (storedRecords) {
                    records = JSON.parse(storedRecords)
                }

                // 查找当天的记录组
                let todayGroup = records.find(group => {
                    const groupDate = new Date(group.timestamp)
                    return dateString === `${groupDate.getFullYear()}-${String(groupDate.getMonth() + 1).padStart(2, '0')}-${String(groupDate.getDate()).padStart(2, '0')}`
                })

                // 如果没有当天的记录组，创建一个
                if (!todayGroup) {
                    todayGroup = {
                        timestamp: Date.now(),
                        routes: []
                    }
                    records.push(todayGroup)
                }

                // 添加新的路线记录
                const newRecord = {
                    timestamp: Date.now(),
                    sessionStartTime: this.routeSessionStartTime,
                    routeInfo: routeInfo,
                    chatHistory: [...this.chatHistory]
                }

                todayGroup.routes.push(newRecord)

                // 按时间倒序排序
                records.sort((a, b) => b.timestamp - a.timestamp)

                // 保存到本地存储
                localStorage.setItem('routeRecords', JSON.stringify(records))

                return true
            } catch (error) {
                console.error('保存路线记录失败:', error)
                return false
            }
        },

        // 设置选中的路线记录
        setSelectedRouteRecord(record) {
            this.selectedRouteRecord = record
        },

        // 加载选中的路线记录
        loadSelectedRouteRecord() {
            if (!this.selectedRouteRecord) return false

            this.chatHistory = [...this.selectedRouteRecord.chatHistory]
            this.routeSessionStartTime = this.selectedRouteRecord.sessionStartTime

            // 清除选中的记录，防止重复加载
            const tempRecord = {...this.selectedRouteRecord }
            this.selectedRouteRecord = null

            return tempRecord
        },

        async loadChatHistory(type) {
            this.loading = true
            try {
                const response = await request.get('/api/chat/completions/history', {
                    params: {
                        type: type || 'general'
                    }
                })

                this.chatHistory = response ?.history || []
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
                    response = await request.post('/api/chat/completions', {
                        messages: [{
                            role: 'user',
                            content: content
                        }],
                        model: 'gemma2:2b',
                        type: type
                    })

                    if (response ?.message) {
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
