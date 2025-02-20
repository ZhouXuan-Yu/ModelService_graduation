<template>
  <div class="image-analysis-assistant">
    <div 
      class="resizer" 
      :class="{ resizing: isResizing }"
      @mousedown="startResize"
    ></div>

    <div class="assistant-content" :style="{ width: assistantWidth + 'px' }">
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
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onUnmounted } from 'vue'
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

// 添加拖动相关的状态和方法
const isResizing = ref(false)
const assistantWidth = ref(400) // 默认宽度
const minWidth = 300 // 最小宽度
const maxWidth = 600 // 最大宽度

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

// 添加拖动相关的方法
const startResize = (e) => {
  isResizing.value = true
  document.addEventListener('mousemove', handleResize)
  document.addEventListener('mouseup', stopResize)
  document.body.classList.add('resizing')
}

const handleResize = (e) => {
  if (!isResizing.value) return
  
  // 获取容器的右边界位置
  const containerRect = document.querySelector('.image-analysis-assistant').getBoundingClientRect()
  const containerRight = containerRect.right
  
  // 计算新的宽度
  const newWidth = containerRight - e.clientX
  
  // 限制宽度在最小值和最大值之间
  assistantWidth.value = Math.min(Math.max(newWidth, minWidth), maxWidth)
}

const stopResize = () => {
  isResizing.value = false
  document.removeEventListener('mousemove', handleResize)
  document.removeEventListener('mouseup', stopResize)
  document.body.classList.remove('resizing')
}

// 组件卸载时清理事件监听
onUnmounted(() => {
  document.removeEventListener('mousemove', handleResize)
  document.removeEventListener('mouseup', stopResize)
})
</script>

<style scoped>
.image-analysis-assistant {
  height: 100%;
  display: flex;
  position: relative;
  background-color: rgba(13, 17, 23, 0.95);
  backdrop-filter: blur(20px);
  box-shadow: 0 0 30px rgba(0, 255, 157, 0.1);
}

.resizer {
  width: 4px;
  background: linear-gradient(180deg, rgba(0, 255, 157, 0.1), rgba(0, 214, 255, 0.1));
  cursor: col-resize;
  position: absolute;
  top: 0;
  bottom: 0;
  left: -2px;
  z-index: 100;
  transition: all 0.3s ease;
}

.resizer:hover,
.resizer.resizing {
  background: linear-gradient(180deg, rgba(0, 255, 157, 0.8), rgba(0, 214, 255, 0.8));
  box-shadow: 0 0 15px rgba(0, 255, 157, 0.5);
  width: 6px;
}

.assistant-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 300px;
  max-width: 600px;
  height: 100%;
  background: rgba(20, 25, 30, 0.7);
  border-left: 1px solid rgba(0, 255, 157, 0.1);
  position: relative;
  backdrop-filter: blur(10px);
}

.assistant-header {
  padding: 16px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(90deg, rgba(13, 17, 23, 0.95), rgba(20, 25, 30, 0.95));
  border-bottom: 1px solid rgba(0, 255, 157, 0.1);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.assistant-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 500;
  color: #ffffff;
  text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
  letter-spacing: 0.5px;
}

.chat-content {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: linear-gradient(135deg, rgba(13, 17, 23, 0.7), rgba(20, 25, 30, 0.7));
}

.message {
  margin-bottom: 20px;
  max-width: 85%;
  opacity: 1;
  transform: translateY(0);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border-radius: 12px;
  overflow: hidden;
}

.message.assistant {
  margin-right: auto;
  background: rgba(0, 255, 157, 0.05);
  border: 1px solid rgba(0, 255, 157, 0.1);
  box-shadow: 0 4px 15px rgba(0, 255, 157, 0.1);
  animation: slideInLeft 0.3s ease;
}

.message.user {
  margin-left: auto;
  background: rgba(0, 214, 255, 0.05);
  border: 1px solid rgba(0, 214, 255, 0.1);
  box-shadow: 0 4px 15px rgba(0, 214, 255, 0.1);
  animation: slideInRight 0.3s ease;
}

.message-content {
  padding: 16px;
  color: rgba(255, 255, 255, 0.9);
  font-size: 14px;
  line-height: 1.6;
}

.input-area {
  padding: 20px;
  background: linear-gradient(0deg, rgba(13, 17, 23, 0.95), rgba(20, 25, 30, 0.7));
  border-top: 1px solid rgba(0, 255, 157, 0.1);
  box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.2);
}

:deep(.el-input__wrapper) {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(0, 255, 157, 0.1);
  box-shadow: none;
  transition: all 0.3s ease;
}

:deep(.el-input__wrapper:hover) {
  border-color: rgba(0, 255, 157, 0.3);
  box-shadow: 0 0 15px rgba(0, 255, 157, 0.1);
}

:deep(.el-input__inner) {
  color: rgba(255, 255, 255, 0.9);
  font-size: 14px;
}

:deep(.el-button) {
  background: linear-gradient(45deg, #00ff9d, #00d6ff);
  border: none;
  color: #0d1117;
  font-weight: 500;
  transition: all 0.3s ease;
}

:deep(.el-button:hover) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0, 255, 157, 0.3);
}

.thinking-dots {
  display: flex;
  gap: 6px;
  padding: 12px;
  justify-content: center;
}

.thinking-dots span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: linear-gradient(45deg, #00ff9d, #00d6ff);
  animation: pulse 1.4s infinite;
  opacity: 0.6;
}

.thinking-dots span:nth-child(2) {
  animation-delay: 0.2s;
}

.thinking-dots span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(0.8);
    opacity: 0.6;
  }
  50% {
    transform: scale(1.2);
    opacity: 1;
  }
}

@keyframes slideInLeft {
  from {
    opacity: 0;
    transform: translateX(-20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes slideInRight {
  from {
    opacity: 0;
    transform: translateX(20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

/* 滚动条样式 */
::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.05);
}

::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #00ff9d, #00d6ff);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, #00ff9d, #00d6ff);
}

/* Markdown 样式优化 */
.markdown-body {
  color: rgba(255, 255, 255, 0.9);
  line-height: 1.6;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3) {
  color: #00ff9d;
  text-shadow: 0 0 10px rgba(0, 255, 157, 0.3);
  border-bottom: 1px solid rgba(0, 255, 157, 0.1);
}

.markdown-body :deep(code) {
  background: rgba(0, 255, 157, 0.1);
  color: #00ff9d;
  border-radius: 4px;
  padding: 2px 6px;
}

.markdown-body :deep(pre) {
  background: rgba(13, 17, 23, 0.8);
  border: 1px solid rgba(0, 255, 157, 0.1);
  border-radius: 8px;
}

.markdown-body :deep(blockquote) {
  border-left: 4px solid #00ff9d;
  background: rgba(0, 255, 157, 0.05);
  margin: 1em 0;
  padding: 1em;
  border-radius: 4px;
}

/* 匹配结果样式 */
.matches {
  margin-top: 12px;
}

.match-item {
  background: rgba(0, 255, 157, 0.05);
  border: 1px solid rgba(0, 255, 157, 0.1);
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.match-item:hover {
  background: rgba(0, 255, 157, 0.1);
  transform: translateX(4px);
  box-shadow: 0 4px 15px rgba(0, 255, 157, 0.2);
}

.match-info {
  display: flex;
  gap: 16px;
  color: rgba(255, 255, 255, 0.9);
  margin-bottom: 8px;
}

.match-colors {
  display: flex;
  gap: 8px;
}

.color-tag {
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 12px;
  color: #ffffff;
  text-shadow: 0 0 4px rgba(0, 0, 0, 0.5);
  background: linear-gradient(45deg, rgba(0, 255, 157, 0.3), rgba(0, 214, 255, 0.3));
}
</style>