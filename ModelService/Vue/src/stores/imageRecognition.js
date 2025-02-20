import { defineStore } from 'pinia'
import { imageRecognitionApi } from '@/api/imageRecognition'

export const useImageRecognitionStore = defineStore('imageRecognition', {
  state: () => ({
    analysisResult: null,
    history: [],
    loading: false
  }),
  
  actions: {
    async analyzeImage(formData) {
      this.loading = true
      try {
        const result = await imageRecognitionApi.analyzeImage(formData)
        this.analysisResult = result
        return result
      } finally {
        this.loading = false
      }
    },
    
    async getHistory() {
      this.loading = true
      try {
        const result = await imageRecognitionApi.getHistory()
        this.history = result
        return result
      } finally {
        this.loading = false
      }
    }
  }
}) 