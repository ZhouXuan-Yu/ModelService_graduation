import axios from 'axios'
import { ElMessage } from 'element-plus'

// 创建 axios 实例
const service = axios.create({
  baseURL: 'http://localhost:8000',  // 确保使用正确的后端地址
  timeout: 180000,  // 增加超时时间到3分钟
  headers: {
    'Accept': 'application/json'
  },
  // 添加重试配置
  retry: 3,
  retryDelay: 1000,
  retryCondition: (error) => {
    return axios.isAxiosError(error) && !error.response;
  }
})

// 请求拦截器
service.interceptors.request.use(
  config => {
    console.log('发送请求:', {
      url: config.url,
      method: config.method,
      data: config.data
    })
    return config
  },
  error => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
service.interceptors.response.use(
  response => {
    console.log('收到原始响应:', response)
    const res = response.data
    
    // 如果响应是直接的内容
    if (res.content !== undefined) {
      return res
    }
    
    // 如果响应包含 success 字段
    if (res.success !== undefined) {
      if (res.success === false) {
        console.error('业务处理失败:', res.message || res.error)
        ElMessage.error(res.message || res.error || '操作失败')
        return Promise.reject(new Error(res.message || res.error || '操作失败'))
      }
      // 如果成功，返回 data 字段
      return res.data || res
    }
    
    // 如果响应包含 error 字段
    if (res.error) {
      console.error('响应包含错误:', res.error)
      ElMessage.error(res.error)
      return Promise.reject(new Error(res.error))
    }
    
    // 其他情况直接返回响应数据
    return res
  },
  error => {
    console.error('响应错误:', error.response || error)
    
    // 处理超时错误
    if (error.code === 'ECONNABORTED') {
      ElMessage.error('请求超时，请重试')
      return Promise.reject(error)
    }
    
    // 处理网络错误
    if (!error.response) {
      ElMessage.error('网络连接失败，请检查网络')
      return Promise.reject(error)
    }
    
    // 处理HTTP错误
    const status = error.response.status
    const errorMsg = error.response.data?.detail || error.message
    
    switch (status) {
      case 422:
        console.error('参数验证失败:', error.response.data)
        ElMessage.error('请求参数验证失败，请检查输入')
        break
      case 500:
        console.error('服务器错误:', errorMsg)
        ElMessage.error(`服务器错误: ${errorMsg}`)
        break
      default:
        console.error(`HTTP错误 ${status}:`, errorMsg)
        ElMessage.error(errorMsg || '请求失败')
    }
    
    return Promise.reject(error)
  }
)

// 添加请求重试拦截器
service.interceptors.response.use(undefined, async (err) => {
  const config = err.config;
  
  if (!config || !config.retry) return Promise.reject(err);
  
  config.__retryCount = config.__retryCount || 0;
  
  if (config.__retryCount >= config.retry) {
    return Promise.reject(err);
  }
  
  config.__retryCount += 1;
  
  const backoff = new Promise(resolve => {
    setTimeout(() => {
      resolve();
    }, config.retryDelay || 1000);
  });
  
  await backoff;
  return service(config);
});

export default service