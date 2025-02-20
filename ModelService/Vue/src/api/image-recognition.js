import request from '@/utils/request'

export const imageRecognitionApi = {
  /**
   * 分析图片
   * @param {FormData} formData - 包含图片文件和分析模式的表单数据
   * @returns {Promise} 返回分析结果
   */
  analyzeImage(formData) {
    console.log('发送图片分析请求:', {
      mode: formData.get('mode'),
      hasFile: formData.has('file')
    })
    
    return request({
      url: '/api/image-recognition/analyze',
      method: 'post',
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 400000
    }).then(response => {
      console.log('图片分析原始响应:', response)
      
      // 检查错误
      if (response.error) {
        console.error('分析错误:', response.error)
        throw new Error(response.error)
      }
      
      // 如果响应是直接的数据对象
      if (response.detected !== undefined && response.persons !== undefined) {
        return response
      }
      
      // 如果响应被包装在 data 字段中
      if (response.data && response.data.detected !== undefined && response.data.persons !== undefined) {
        return response.data
      }
      
      // 如果响应被包装在 success 和 data 字段中
      if (response.success && response.data && response.data.detected !== undefined) {
        return response.data
      }
      
      console.error('响应数据格式不正确:', response)
      throw new Error('响应数据格式不正确')
    }).catch(error => {
      console.error('请求失败:', error)
      throw error
    })
  },

  /**
   * 健康检查
   * @returns {Promise} 返回健康状态
   */
  healthCheck() {
    return request({
      url: '/api/image-recognition/health',
      method: 'get'
    })
  }
} 