<template>
  <div class="chat-panel">
    <el-card class="chat-card">
      <template #header>
        <div class="chat-header">
          <span>智能助手</span>
          <el-button-group>
            <el-button @click="clearHistory">
              <el-icon><Delete /></el-icon>
            </el-button>
          </el-button-group>
        </div>
      </template>
      
      <div class="chat-content" ref="chatContent">
        <!-- 聊天历史 -->
        <div class="chat-messages">
          <div v-for="(message, index) in chatHistory" 
               :key="index" 
               class="message"
               :class="message.type">
            <div class="message-content">
              <div class="message-text" v-html="formatMessage(message.content)"></div>
              <div class="message-time">{{ formatTime(message.timestamp) }}</div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 输入区域 -->
      <div class="chat-input">
        <el-input
          v-model="inputMessage"
          type="textarea"
          :rows="3"
          placeholder="请输入查询，例如：'找到穿绿色上衣的女性'"
          @keyup.enter.exact="handleMessage"
        />
        <div class="input-actions">
          <el-button-group>
            <el-button 
              type="primary" 
              @click="handleMessage"
              :disabled="!inputMessage.trim()"
            >
              发送
            </el-button>
          </el-button-group>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { Delete } from '@element-plus/icons-vue'
import { useAnalysisStore } from '@/stores/analysis'
import { parseQuery, generateResultDescription } from '@/utils/queryParser'
import dayjs from 'dayjs'
import { chatApi } from '@/api/chat'

// Props
const props = defineProps({
  analysisResult: {
    type: Object,
    default: null
  }
})

// Emits
const emit = defineEmits(['query'])

// Store
const analysisStore = useAnalysisStore()

// Refs
const chatContent = ref(null)
const inputMessage = ref('')
const chatHistory = ref([
  {
    type: 'system',
    content: '你好！我是智能助手，可以帮你查找和分析图片中的人物。上传图片并分析后，你可以问我类似这样的问题：<br>- 找到穿绿色上衣的女性<br>- 帮我找年龄小于30岁的人<br>- 谁穿着蓝色的衣服？',
    timestamp: new Date()
  }
])

// Methods
const handleMessage = async (message) => {
  if (!inputMessage.value.trim()) return
  
  // 添加用户消息
  addMessage('user', inputMessage.value)
  
  // 处理查询
  const query = inputMessage.value
  inputMessage.value = ''
  
  if (!props.analysisResult) {
    addMessage('system', '请先上传并分析图片')
    return
  }
  
  try {
    // 调用本地大模型进行对话
    const response = await chatApi.sendMessage({
      messages: [
        {
          role: 'system',
          content: '你是一个图像分析助手，可以帮助用户查找和分析图片中的人物。请根据用户的描述，找到匹配的人物并生成自然的回复。'
        },
        {
          role: 'user',
          content: query
        }
      ],
      type: 'image_analysis'
    })
    
    // 解析查询条件
    const conditions = parseQuery(query)
    const matches = analysisStore.findMatchingPersons(conditions)
    
    // 生成回复
    let reply = ''
    if (matches.length > 0) {
      reply = generateResultDescription(matches, conditions)
      // 高亮匹配的人物
      matches.forEach((match, index) => {
        setTimeout(() => {
          analysisStore.setActivePerson(match.id)
        }, index * 1000)
      })
    } else {
      reply = '抱歉，我没有找到符合描述的人物。请尝试其他描述方式。'
    }
    
    // 添加系统回复
    addMessage('system', reply)
    
    // 发送查询事件
    emit('query', { query, matches })
    
  } catch (error) {
    console.error('处理查询失败:', error)
    addMessage('system', '抱歉，处理查询时出现错误，请重试。')
  }
}

const addMessage = (type, content) => {
  chatHistory.value.push({
    type,
    content,
    timestamp: new Date()
  })
  
  // 滚动到底部
  nextTick(() => {
    if (chatContent.value) {
      chatContent.value.scrollTop = chatContent.value.scrollHeight
    }
  })
}

const clearHistory = () => {
  chatHistory.value = [chatHistory.value[0]] // 保留系统欢迎消息
}

const formatMessage = (text) => {
  // 处理换行
  text = text.replace(/\n/g, '<br>')
  
  // 处理颜色关键词高亮
  Object.entries(COLOR_EN_TO_CN).forEach(([en, cn]) => {
    const pattern = new RegExp(cn, 'g')
    text = text.replace(pattern, `<span class="color-keyword" style="color: ${en}">${cn}</span>`)
  })
  
  return text
}

const formatTime = (timestamp) => {
  return dayjs(timestamp).format('HH:mm')
}

// 监听分析结果变化
watch(() => props.analysisResult, (newResult) => {
  if (newResult) {
    addMessage('system', `分析完成，检测到 ${newResult.num_faces} 个人物。你可以开始查询了！`)
  }
})
</script>

<style scoped>
.chat-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.chat-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chat-content {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background-color: #f5f7fa;
}

.chat-messages {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.message {
  max-width: 80%;
  margin: 8px 0;
  padding: 12px;
  border-radius: 8px;
  word-break: break-word;
}

.message.user {
  align-self: flex-end;
  background-color: var(--el-color-primary-light-9);
  margin-left: 20%;
}

.message.system {
  align-self: flex-start;
  background-color: #f4f4f5;
  margin-right: 20%;
}

.color-keyword {
  font-weight: 500;
  padding: 0 2px;
}

.message-text {
  word-break: break-word;
  line-height: 1.4;
}

.message-time {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
  text-align: right;
}

.chat-input {
  padding: 20px;
  border-top: 1px solid var(--el-border-color-light);
  background-color: white;
}

.input-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 12px;
}

:deep(.el-textarea__inner) {
  resize: none;
}
</style> 