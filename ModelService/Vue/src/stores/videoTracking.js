import { defineStore } from 'pinia'
import { videoTrackingApi } from '@/api/videoTracking'

export const useVideoTrackingStore = defineStore('videoTracking', {
  state: () => ({
    trackingResult: null,
    history: [],
    loading: false
  }),
  
  actions: {
    async startTracking(formData) {
      this.loading = true
      try {
        const result = await videoTrackingApi.startTracking(formData)
        this.trackingResult = result
        return result
      } finally {
        this.loading = false
      }
    },
    
    async getTrackingStatus(trackingId) {
      try {
        const result = await videoTrackingApi.getTrackingStatus(trackingId)
        this.trackingResult = result
        return result
      } catch (error) {
        console.error('获取追踪状态失败:', error)
        throw error
      }
    },
    
    async getHistory() {
      this.loading = true
      try {
        const result = await videoTrackingApi.getHistory()
        this.history = result
        return result
      } finally {
        this.loading = false
      }
    }
  }
}) 