<template>
  <div class="chat-assistant">
    <div class="chat-container">
      <!-- 使用 v-loading 指令替代 el-loading 组件 -->
      <div v-loading="store.loading">
        <!-- 错误提示 -->
        <el-alert
          v-if="store.error"
          :title="store.error"
          type="error"
          show-icon
        />
        
        <!-- 聊天消息 -->
        <div class="chat-messages" ref="messagesContainer">
          <div v-for="(message, index) in store.chatHistory" 
               :key="index"
               :class="['message', message.role]">
            <div class="message-content">
              <div v-if="message.role === 'assistant'" class="markdown-content">
                <div v-html="marked(message.content, { breaks: true })"></div>
              </div>
              <div v-else>{{ message.content }}</div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 输入框 -->
      <div class="chat-input">
        <el-input
          v-model="userInput"
          type="textarea"
          :rows="3"
          placeholder="请输入您的问题..."
          @keyup.enter.ctrl="sendMessage"
        />
        <el-button 
          type="primary" 
          @click="sendMessage"
          :loading="loading"
        >
          发送
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch } from 'vue'
import { useRoute } from 'vue-router'
import { marked } from 'marked'
import { useMainStore } from '@/stores'
import hljs from 'highlight.js'
import 'highlight.js/styles/github-dark.css'

// 配置 marked
marked.setOptions({
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value
    }
    return code
  },
  breaks: true,
  gfm: true
})

const store = useMainStore()
const route = useRoute()
const userInput = ref('')
const loading = ref(false)
const messagesContainer = ref(null)

// 根据路由获取当前聊天类型
const getChatType = () => {
  const path = route.path
  if (path.includes('route-planning')) return 'route'
  if (path.includes('image-recognition')) return 'image'
  if (path.includes('video-tracking')) return 'video'
  return 'general'
}

// 发送消息
const sendMessage = async () => {
  if (!userInput.value.trim() || loading.value) return
  
  const message = userInput.value
  userInput.value = ''
  loading.value = true
  
  try {
    await store.sendChatMessage({
      type: getChatType(),
      content: message
    })
  } catch (error) {
    console.error('发送消息失败:', error)
  } finally {
    loading.value = false
    scrollToBottom()
  }
}

// 滚动到底部
const scrollToBottom = async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

// 监听聊天历史变化
watch(() => store.chatHistory, () => {
  scrollToBottom()
}, { deep: true })

onMounted(async () => {
  // 加载历史消息
  await store.loadChatHistory(getChatType())
})
</script>

<style scoped>
.chat-assistant {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--el-bg-color);
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px;
  gap: 20px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
}

.message {
  margin-bottom: 20px;
  display: flex;
}

.message.user {
  justify-content: flex-end;
}

.message-content {
  max-width: 80%;
  padding: 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.1);
}

.message.assistant .message-content {
  background: rgba(0, 255, 157, 0.1);
}

.chat-input {
  display: flex;
  gap: 12px;
}

.chat-input .el-input {
  flex: 1;
}

.markdown-content :deep(pre) {
  background: rgba(0, 0, 0, 0.2);
  padding: 15px;
  border-radius: 4px;
  overflow-x: auto;
  margin: 10px 0;
}

.markdown-content :deep(code) {
  font-family: monospace;
  color: var(--el-color-primary);
  padding: 2px 4px;
  border-radius: 3px;
  font-size: 0.9em;
}

.markdown-content :deep(p) {
  margin: 8px 0;
  line-height: 1.6;
}

.markdown-content :deep(ul), 
.markdown-content :deep(ol) {
  padding-left: 20px;
  margin: 8px 0;
}

.markdown-content :deep(blockquote) {
  border-left: 4px solid var(--el-color-primary);
  margin: 8px 0;
  padding: 0 12px;
  color: rgba(255, 255, 255, 0.7);
}
</style> 