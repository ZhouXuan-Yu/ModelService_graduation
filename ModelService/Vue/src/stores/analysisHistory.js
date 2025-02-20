import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAnalysisHistoryStore = defineStore('analysisHistory', () => {
  // 存储所有分析历史
  const analysisHistory = ref([])
  
  // 当前活跃的分析结果ID
  const activeAnalysisId = ref(null)
  
  // 获取当前活跃的分析结果
  const activeAnalysis = computed(() => {
    return analysisHistory.value.find(item => item.id === activeAnalysisId.value)
  })
  
  // 添加新的分析结果
  function addAnalysis(result) {
    const analysisItem = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      result,
      imageUrl: result.result_image_url
    }
    analysisHistory.value.push(analysisItem)
    activeAnalysisId.value = analysisItem.id
    return analysisItem.id
  }
  
  // 设置当前活跃的分析结果
  function setActiveAnalysis(id) {
    activeAnalysisId.value = id
  }
  
  // 获取指定ID的分析结果
  function getAnalysisById(id) {
    return analysisHistory.value.find(item => item.id === id)
  }
  
  // 获取所有分析历史的摘要信息
  const analysisSummaries = computed(() => {
    return analysisHistory.value.map(item => ({
      id: item.id,
      timestamp: item.timestamp,
      numFaces: item.result.num_faces,
      isActive: item.id === activeAnalysisId.value
    }))
  })
  
  // 清除历史记录
  function clearHistory() {
    analysisHistory.value = []
    activeAnalysisId.value = null
  }
  
  return {
    analysisHistory,
    activeAnalysisId,
    activeAnalysis,
    analysisSummaries,
    addAnalysis,
    setActiveAnalysis,
    getAnalysisById,
    clearHistory
  }
})