export const parseAIResponse = (response) => {
  try {
    // 尝试解析JSON响应
    const parsed = JSON.parse(response)
    return {
      routePlan: parsed,
      displayText: JSON.stringify(parsed, null, 2)
    }
  } catch (e) {
    // 如果不是JSON，返回原始文本
    return {
      routePlan: null,
      displayText: response
    }
  }
}

export const formatMessage = (message) => {
  if (typeof message === 'string') {
    return message
  }
  return JSON.stringify(message, null, 2)
} 