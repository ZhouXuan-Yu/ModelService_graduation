import request from '@/utils/request'

export const videoTrackingApi = {
  /**
   * 分析视频
   * @param {FormData} formData - 包含视频和描述的表单数据
   * @returns {Promise} 返回分析结果
   */
  analyzeVideo(formData) {
    return request({
      url: '/api/video-tracking/analyze',
      method: 'post',
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 300000 // 5分钟超时
    })
  },

  /**
   * 健康检查
   * @returns {Promise} 返回健康状态
   */
  healthCheck() {
    return request({
      url: '/api/video/health',
      method: 'get'
    })
  }
} 