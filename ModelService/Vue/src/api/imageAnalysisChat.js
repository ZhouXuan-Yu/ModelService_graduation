import request from '@/utils/request'

export const imageAnalysisChatApi = {
  /**
   * 发送消息到本地大模型
   * @param {string} message - 用户消息
   * @param {Object} analysisData - 当前分析结果
   * @returns {Promise}
   */
  sendMessage(message, analysisData = null) {
    console.log('=== 开始发送聊天请求 ===')
    console.log('用户消息:', message)
    console.log('分析数据:', analysisData)

    // 确保 analysisData 的格式正确
    const formattedAnalysisData = analysisData ? {
      currentAnalysis: {
        persons: Array.isArray(analysisData.persons) ? analysisData.persons.map((person, index) => ({
          id: index + 1,
          age: parseFloat(person.age || 0),
          age_confidence: parseFloat(person.age_confidence || 1.0),
          gender: person.gender || "unknown",
          gender_confidence: parseFloat(person.gender_confidence || 0),
          upper_color: person.upper_color || "unknown",
          upper_color_confidence: parseFloat(person.upper_color_confidence || 0),
          lower_color: person.lower_color || "unknown",
          lower_color_confidence: parseFloat(person.lower_color_confidence || 0),
          bbox: Array.isArray(person.bbox) ? person.bbox.map(Number) : [0, 0, 0, 0]
        })) : [],
        detected: parseInt(analysisData.detected || 0)
      },
      analysisHistory: []
    } : { currentAnalysis: { persons: [], detected: 0 }, analysisHistory: [] }

    console.log('格式化后的请求数据:', formattedAnalysisData)

    const requestData = {
      messages: [{
        role: "user",
        content: message.toString()
      }],
      model: "gemma2:2b",
      temperature: 0.7,
      context: formattedAnalysisData
    }

    console.log('发送到后端的完整数据:', JSON.stringify(requestData, null, 2))

    return request({
      url: '/api/image-analysis-chat/completions',
      method: 'post',
      data: requestData,
      timeout: 180000,  // 增加超时时间到3分钟
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(response => {
      console.log('收到后端响应:', response)
      return response
    }).catch(error => {
      console.error('请求失败:', error)
      throw error
    })
  }
}