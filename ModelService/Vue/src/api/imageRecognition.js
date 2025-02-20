import request from './request'

export const imageRecognitionApi = {
  // 修改为使用chat接口
  analyzeImage(formData) {
    return request.post('/chat', {
      messages: [{
        role: 'user',
        content: formData.get('description')
      }],
      model: 'gemma2:2b',
      mode: 'image_recognition',
      image: formData.get('image')
    })
  },
  
  // 暂时使用chat接口获取历史
  getHistory() {
    return request.get('/chat')
  }
} 