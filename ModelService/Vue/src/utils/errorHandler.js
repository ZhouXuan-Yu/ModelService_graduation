// 新建错误处理工具
export const handleApiError = (error) => {
  if (error.code === 'ERR_NETWORK') {
    return '网络连接失败，请检查服务器是否正常运行'
  }
  if (error.response) {
    switch (error.response.status) {
      case 401:
        return '未授权，请登录'
      case 403:
        return '拒绝访问'
      case 404:
        return '请求地址不存在'
      case 500:
        return '服务器内部错误'
      default:
        return `请求失败 (${error.response.status})`
    }
  }
  return error.message
} 