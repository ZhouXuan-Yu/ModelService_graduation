<template>
  <div class="image-analysis-assistant">
    <div class="assistant-header">
      <h3>图片分析助手</h3>
      <div class="header-actions">
        <el-dropdown @command="handleHistorySelect">
          <el-button type="text">
            历史记录
            <el-icon class="el-icon--right"><arrow-down /></el-icon>
          </el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item 
                v-for="item in analysisHistoryStore.analysisSummaries" 
                :key="item.id"
                :command="item.id"
                :class="{ 'active': item.isActive }"
              >
                {{ formatTime(item.timestamp) }} ({{ item.numFaces }}人)
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
        <el-button type="text" @click="clearHistory">
          <el-icon><Delete /></el-icon>
          清空历史
        </el-button>
      </div>
    </div>

    <div class="chat-content" ref="chatContent">
      <div v-for="(msg, index) in chatHistory" 
           :key="index" 
           :class="['message', msg.role]">
        <div class="message-content markdown-body" v-html="formatMessage(msg)"></div>
        <div v-if="msg.matches?.length" class="matches">
          <div v-for="match in msg.matches" 
               :key="match.id"
               class="match-item"
               @click="highlightPerson(match.id)">
            <div class="match-info">
              <span>性别: {{ match.gender }}</span>
              <span>年龄: {{ match.age }}</span>
            </div>
            <div class="match-colors">
              <span class="color-tag" :style="getColorStyle(match.upper_color)">
                上衣: {{ translateColor(match.upper_color) }}
              </span>
              <span class="color-tag" :style="getColorStyle(match.lower_color)">
                下装: {{ translateColor(match.lower_color) }}
              </span>
            </div>
          </div>
        </div>
        <div class="message-time">{{ formatTime(msg.timestamp) }}</div>
      </div>
      <div v-if="loading" class="message assistant thinking">
        <div class="thinking-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>

    <div class="input-area">
      <el-input
        v-model="inputMessage"
        :placeholder="inputPlaceholder"
        @keyup.enter="sendMessage"
        :disabled="loading"
        clearable
      >
        <template #append>
          <el-button @click="sendMessage" :loading="loading">发送</el-button>
        </template>
      </el-input>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { Delete, ArrowDown } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useAnalysisHistoryStore } from '@/stores/analysisHistory'
import { imageAnalysisChatApi } from '@/api/imageAnalysisChat'
import { translateColor, getColorStyle } from '@/utils/colorMapping'
import dayjs from 'dayjs'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

const props = defineProps({
  analysisResult: {
    type: Object,
    default: null
  }
})

const emit = defineEmits(['query'])

// Store
const analysisHistoryStore = useAnalysisHistoryStore()

// Refs
const chatContent = ref(null)
const inputMessage = ref('')
const chatHistory = ref([
  {
    role: 'assistant',
    content: '你好！我是图片分析助手。我可以帮你分析图片中的人物信息。<br>上传并分析图片后，你可以问我类似这样的问题：<br>- 帮我找穿黄色上衣的男性<br>- 有没有年龄小于30岁的女性<br>- 谁穿着蓝色衣服？',
    timestamp: new Date()
  }
])

const loading = ref(false)
const isProcessing = ref(false)

// Methods
const sendMessage = async () => {
  if (!inputMessage.value.trim() || isProcessing.value) return
  
  const message = inputMessage.value.trim()
  inputMessage.value = ''
  isProcessing.value = true
  
  // 添加用户消息到聊天记录
  addMessage({
    role: 'user',
    content: message,
    timestamp: new Date()
  })
  
  try {
    // 获取当前活跃的分析结果
    const currentAnalysis = props.analysisResult || analysisHistoryStore.activeAnalysis?.result

    if (!currentAnalysis) {
      throw new Error('当前没有可用的图片分析结果')
    }

    // 发送到本地大模型
    const response = await imageAnalysisChatApi.sendMessage(message, currentAnalysis)
    
    console.log('聊天响应:', response)
    
    if (response.content) {
      // 添加助手回复到聊天记录
      addMessage({
        role: 'assistant',
        content: response.content,
        timestamp: new Date()
      })
    } else if (response.data && response.data.content) {
      addMessage({
        role: 'assistant',
        content: response.data.content,
        timestamp: new Date()
      })
    } else {
      throw new Error('无效的响应格式')
    }
  } catch (error) {
    console.error('处理消息失败:', error)
    addMessage({
      role: 'error',
      content: error.response?.data?.detail || error.message || '消息处理失败，请重试',
      timestamp: new Date()
    })
  } finally {
    isProcessing.value = false
  }
}

const addMessage = (message) => {
  chatHistory.value.push({
    ...message,
    matches: message.matches || []
  })
  
  // 滚动到底部
  nextTick(() => {
    if (chatContent.value) {
      chatContent.value.scrollTop = chatContent.value.scrollHeight
    }
  })
}

const handleHistorySelect = (id) => {
  analysisHistoryStore.setActiveAnalysis(id)
  const analysis = analysisHistoryStore.getAnalysisById(id)
  if (analysis) {
    addMessage('system', `切换到 ${formatTime(analysis.timestamp)} 的分析结果，共检测到 ${analysis.result.num_faces} 个人物。`)
  }
}

const clearHistory = () => {
  ElMessageBox.confirm('确定要清空所有历史记录吗？', '提示', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(() => {
    analysisHistoryStore.clearHistory()
    chatHistory.value = [chatHistory.value[0]] // 保留欢迎消息
    ElMessage.success('历史记录已清空')
  }).catch(() => {})
}

const formatMessage = (msg) => {
  if (!msg || !msg.content) return ''
  // 使用 marked 将 Markdown 转换为 HTML，并使用 DOMPurify 清理
  const cleanHtml = DOMPurify.sanitize(marked(msg.content))
  return cleanHtml
}

const formatTime = (timestamp) => {
  return dayjs(timestamp).format('HH:mm')
}

const highlightPerson = (id) => {
  emit('query', { matches: [{ id }] })
}

// 监听分析结果变化
watch(() => props.analysisResult, (newResult) => {
  if (newResult) {
    // 添加到历史记录
    const analysisId = analysisHistoryStore.addAnalysis(newResult)
    addMessage('assistant', `图片分析完成，检测到 ${newResult.num_faces} 个人物。
你可以：
1. 询问具体人物的信息（如："找到穿红色上衣的人"）
2. 比较与之前图片的异同（如："这张图片比上一张多了几个人？"）
3. 获取更详细的分析报告（如："帮我总结一下这张图片的特点"）`)
  }
})

// 计算属性
const inputPlaceholder = computed(() => {
  return analysisHistoryStore.activeAnalysis
    ? '请输入你的问题，例如："找到穿红色上衣的人"'
    : '请先上传并分析图片...'
})

// 修改 notifyAnalysisComplete 方法
const notifyAnalysisComplete = async (result) => {
  if (!result) {
    console.error('Invalid analysis data:', result)
    return
  }

  try {
    console.log('分析助手收到分析结果:', result)

    // 格式化分析结果
    const formattedResult = {
      detected: result.detected || 0,
      persons: Array.isArray(result.persons) ? result.persons.map((person, index) => ({
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
      })) : []
    }

    // 更新聊天历史
    addMessage({
      role: 'assistant',
      content: `图片分析完成，检测到 ${formattedResult.detected} 个人物。
分析结果：
${formattedResult.persons.map((person, index) => `
人物${index + 1}：
- 性别：${person.gender}（置信度：${(person.gender_confidence * 100).toFixed(1)}%）
- 年龄：${person.age.toFixed(1)}岁（置信度：${(person.age_confidence * 100).toFixed(1)}%）
- 上衣：${person.upper_color}（置信度：${(person.upper_color_confidence * 100).toFixed(1)}%）
- 下装：${person.lower_color}（置信度：${(person.lower_color_confidence * 100).toFixed(1)}%）
`).join('\n')}

你可以开始询问了，例如：
- "找到穿红色上衣的人"
- "有没有年龄小于30岁的女性"
- "谁穿着蓝色衣服？"`,
      timestamp: new Date()
    })

  } catch (error) {
    console.error('处理分析结果失败:', error)
    ElMessage.error('处理分析结果失败')
  }
}

// 对外暴露方法
defineExpose({
  notifyAnalysisComplete
})
</script>

<style scoped>
.image-analysis-assistant {
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: var(--el-bg-color);
  color: var(--el-text-color-primary);
}

.assistant-header {
  padding: 12px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #141414;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.header-actions {
  display: flex;
  gap: 12px;
}

.assistant-header h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 500;
}

.chat-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  background-color: var(--el-bg-color);
}

.message {
  margin-bottom: 16px;
  max-width: 85%;
  opacity: 1;
  transform: translateY(0);
  transition: all 0.3s ease;
}

.message.assistant {
  margin-right: auto;
  animation: slideIn 0.3s ease;
}

.message.user {
  margin-left: auto;
  animation: slideIn 0.3s ease;
}

.message-content {
  padding: 12px;
  border-radius: 8px;
  background-color: rgba(255, 255, 255, 0.05);
  line-height: 1.5;
  word-break: break-word;
}

.message.user .message-content {
  background-color: var(--el-color-primary-light-9);
  color: var(--el-color-primary-dark-2);
}

.message-time {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  margin-top: 4px;
}

.matches {
  margin-top: 12px;
}

.match-item {
  padding: 12px;
  margin-bottom: 8px;
  border-radius: 6px;
  background-color: rgba(255, 255, 255, 0.03);
  cursor: pointer;
  transition: background-color 0.2s;
}

.match-item:hover {
  background-color: rgba(255, 255, 255, 0.08);
}

.match-info {
  display: flex;
  gap: 16px;
  margin-bottom: 8px;
}

.match-colors {
  display: flex;
  gap: 8px;
}

.color-tag {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  color: #fff;
  text-shadow: 0 0 2px rgba(0, 0, 0, 0.5);
}

.input-area {
  padding: 16px;
  background-color: rgba(255, 255, 255, 0.02);
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

:deep(.el-input__wrapper) {
  background-color: rgba(255, 255, 255, 0.04);
}

:deep(.el-input__inner) {
  color: var(--el-text-color-primary);
}

:deep(.el-input__inner::placeholder) {
  color: var(--el-text-color-secondary);
}

:deep(.el-dropdown-menu__item.active) {
  color: var(--el-color-primary);
  font-weight: 500;
}

/* 添加思考中的动画样式 */
.thinking {
  padding: 12px;
  margin-bottom: 16px;
  max-width: 100px;
}

.thinking-dots {
  display: flex;
  gap: 4px;
  align-items: center;
  justify-content: center;
}

.thinking-dots span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: var(--el-color-primary);
  animation: thinking 1.4s infinite;
  opacity: 0.4;
}

.thinking-dots span:nth-child(2) {
  animation-delay: 0.2s;
}

.thinking-dots span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes thinking {
  0%, 100% {
    transform: translateY(0);
    opacity: 0.4;
  }
  50% {
    transform: translateY(-4px);
    opacity: 1;
  }
}

/* 优化消息样式 */
@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 移除加载遮罩相关样式 */
:deep(.el-loading-mask) {
  display: none !important;
}

/* 添加 Markdown 样式 */
.markdown-body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  line-height: 1.6;
}

.markdown-body :deep(h1) {
  font-size: 2em;
  margin: 0.67em 0;
  border-bottom: 1px solid var(--el-border-color-light);
  padding-bottom: 0.3em;
}

.markdown-body :deep(h2) {
  font-size: 1.5em;
  margin: 0.83em 0;
  border-bottom: 1px solid var(--el-border-color-light);
  padding-bottom: 0.3em;
}

.markdown-body :deep(h3) {
  font-size: 1.17em;
  margin: 1em 0;
}

.markdown-body :deep(ul), .markdown-body :deep(ol) {
  padding-left: 2em;
  margin: 1em 0;
}

.markdown-body :deep(li) {
  margin: 0.5em 0;
}

.markdown-body :deep(code) {
  background-color: var(--el-bg-color-page);
  padding: 0.2em 0.4em;
  border-radius: 3px;
  font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 85%;
}

.markdown-body :deep(pre) {
  background-color: var(--el-bg-color-page);
  padding: 1em;
  border-radius: 6px;
  overflow-x: auto;
}

.markdown-body :deep(blockquote) {
  margin: 1em 0;
  padding: 0 1em;
  color: var(--el-text-color-secondary);
  border-left: 0.25em solid var(--el-border-color);
}

.markdown-body :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
}

.markdown-body :deep(th), .markdown-body :deep(td) {
  border: 1px solid var(--el-border-color);
  padding: 6px 13px;
}

.markdown-body :deep(th) {
  background-color: var(--el-bg-color-page);
}

.markdown-body :deep(img) {
  max-width: 100%;
  height: auto;
}

.markdown-body :deep(p) {
  margin: 1em 0;
}

.markdown-body :deep(hr) {
  height: 0.25em;
  border: 0;
  background-color: var(--el-border-color);
  margin: 24px 0;
}

.message.assistant .message-content {
  background-color: var(--el-bg-color-overlay);
  border: 1px solid var(--el-border-color-light);
}

.message.user .message-content {
  background-color: var(--el-color-primary-light-9);
}
</style>