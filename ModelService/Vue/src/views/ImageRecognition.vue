<template>
  <div class="image-recognition">
    <el-container class="main-container">
      <!-- 左侧图片分析区域 -->
      <el-main class="analysis-panel">
        <el-card class="upload-card" v-if="!hasResults">
          <template #header>
            <div class="card-header">
              <span>上传图片</span>
            </div>
          </template>
          <div class="upload-container">
            <el-upload
              class="image-uploader"
              :show-file-list="false"
              :before-upload="beforeUpload"
              :http-request="customUpload"
              accept="image/*"
            >
              <div class="upload-content" v-loading="loading">
                <img v-if="imageUrl" :src="imageUrl" class="preview-image" />
                <el-icon v-else class="upload-icon"><Plus /></el-icon>
              </div>
            </el-upload>
            
            <div class="toolbar">
              <el-button 
                type="primary" 
                @click="analyzeImage"
                :loading="loading"
                :disabled="!imageUrl"
                class="analyze-btn"
              >
                {{ loading ? '分析中...' : '开始分析' }}
              </el-button>
              <el-button-group v-if="hasResults">
                <el-button @click="zoomIn">
                  <el-icon><ZoomIn /></el-icon>
                </el-button>
                <el-button @click="zoomOut">
                  <el-icon><ZoomOut /></el-icon>
                </el-button>
                <el-button @click="resetZoom">
                  <el-icon><FullScreen /></el-icon>
                </el-button>
              </el-button-group>
            </div>

            <!-- 添加进度提示 -->
            <el-progress 
              v-if="loading"
              :percentage="analysisProgress"
              :format="progressFormat"
              class="progress-bar"
            />

            <!-- 在上传区域添加分析模式选择 -->
            <div class="analysis-mode-section">
              <h3>分析模式</h3>
              <el-radio-group v-model="analysisMode" class="mode-selector">
                <el-radio-button label="normal">
                  <el-tooltip content="仅使用本地模型，分析速度更快" placement="top">
                    <span>普通分析</span>
                  </el-tooltip>
                </el-radio-button>
                <el-radio-button label="enhanced">
                  <el-tooltip content="同时使用本地模型和Qwen-VL模型，分析更全面" placement="top">
                    <span>加强分析</span>
                  </el-tooltip>
                </el-radio-button>
              </el-radio-group>
            </div>
          </div>
        </el-card>

        <!-- 分析结果展示 -->
        <el-card v-if="analysisResult" class="result-card">
          <template #header>
            <div class="result-header">
              <span>分析结果</span>
              <div class="result-actions">
                <el-button-group>
                  <el-button @click="exportResults">
                    <el-icon><Download /></el-icon>
                  </el-button>
                  <el-button @click="shareResults">
                    <el-icon><Share /></el-icon>
                  </el-button>
                </el-button-group>
              </div>
            </div>
          </template>
          
          <div class="result-content" ref="resultContainer">
            <div class="result-image-container" 
                 :style="{ transform: `scale(${zoomLevel})` }"
                 @mousedown="startPan"
                 @mousemove="pan"
                 @mouseup="endPan"
                 @mouseleave="endPan">
              <img 
                v-if="imageUrl"
                :src="imageUrl"
                alt="分析结果"
                class="result-image"
                :style="{ transform: `translate(${panX}px, ${panY}px)` }"
                @load="onImageLoad"
              />
              <!-- 人物标注层 -->
              <div class="annotations-layer">
                <div v-for="(person, index) in analysisResult?.persons" 
                     :key="index"
                     class="person-annotation"
                     :class="{ active: activePersonId === index }"
                     :style="getAnnotationStyle(person.bbox)"
                     @click="handlePersonClick(index)">
                  <div class="annotation-label">
                    {{ index + 1 }}
                  </div>
                </div>
              </div>
            </div>
            
            <div class="result-details">
              <h3>检测到 {{ analysisResult?.detected || 0 }} 个人物</h3>
              <el-collapse v-model="activeNames">
                <el-collapse-item 
                  v-for="(person, index) in analysisResult?.persons" 
                  :key="index"
                  :title="'人物 ' + (index + 1)"
                  :name="index"
                >
                  <div class="person-info">
                    <el-descriptions :column="1" border>
                      <el-descriptions-item label="年龄">
                        {{ person.age?.toFixed(1) || '未知' }} 岁
                        <el-tag size="small" type="info">
                          置信度: {{ (person.age_confidence * 100).toFixed(1) }}%
                        </el-tag>
                      </el-descriptions-item>
                      <el-descriptions-item label="性别">
                        {{ translateGender(person.gender) || '未知' }}
                        <el-tag size="small" type="info">
                          置信度: {{ (person.gender_confidence * 100).toFixed(1) }}%
                        </el-tag>
                      </el-descriptions-item>
                      <el-descriptions-item label="上衣颜色">
                        <span :style="getColorStyle(person.upper_color)" class="color-tag">
                          {{ translateColor(person.upper_color || '未知') }}
                        </span>
                        <el-tag size="small" type="info">
                          置信度: {{ (person.upper_color_confidence * 100).toFixed(1) }}%
                        </el-tag>
                      </el-descriptions-item>
                      <el-descriptions-item label="下衣颜色">
                        <span :style="getColorStyle(person.lower_color)" class="color-tag">
                          {{ translateColor(person.lower_color || '未知') }}
                        </span>
                        <el-tag size="small" type="info">
                          置信度: {{ (person.lower_color_confidence * 100).toFixed(1) }}%
                        </el-tag>
                      </el-descriptions-item>
                    </el-descriptions>
                  </div>
                </el-collapse-item>
              </el-collapse>
            </div>
          </div>
        </el-card>
      </el-main>
    </el-container>

    <!-- 修改聊天区域，添加 ref -->
    <div class="chat-section" :class="{ 'has-results': hasResults }">
      <ImageAnalysisAssistant
        ref="assistantRef"
        :analysis-result="analysisResult"
        @query="handleQuery"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { Plus, ZoomIn, ZoomOut, FullScreen, Download, Share } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { useAnalysisStore } from '@/stores/analysis'
import { useAnalysisHistoryStore } from '@/stores/analysisHistory'
import { imageRecognitionApi } from '@/api/image-recognition'
import { parseQuery, generateResultDescription } from '@/utils/queryParser'
import { COLOR_EN_TO_CN } from '@/utils/colorMapping'
import { imageAnalysisChatApi } from '@/api/imageAnalysisChat'
import ImageAnalysisAssistant from '@/components/ImageAnalysisAssistant.vue'

const emit = defineEmits(['analysis-complete'])

// 状态管理
const analysisStore = useAnalysisStore()
const analysisHistoryStore = useAnalysisHistoryStore()
const imageUrl = ref('')
const loading = ref(false)
const analysisResult = ref(null)
const activePersonId = computed(() => analysisStore.activePersonId)
const resultImageUrl = ref('')
const activeNames = ref([0])
const imageFile = ref(null)
const isAnalyzing = ref(false)

// 缩放和平移相关
const zoomLevel = ref(1)
const panX = ref(0)
const panY = ref(0)
const isPanning = ref(false)
const lastX = ref(0)
const lastY = ref(0)

// 添加分屏相关的状态和方法
const isResizing = ref(false)
const startX = ref(0)

// 添加分析模式状态
const analysisMode = ref('normal')

// 添加进度相关的状态和方法
const analysisProgress = ref(0)
const progressInterval = ref(null)

const progressFormat = (percentage) => {
  return `${percentage}% 处理中...`
}

// 添加 assistantRef
const assistantRef = ref(null)

// 添加聊天相关的状态
const chatInput = ref('')
const chatMessages = ref([])
const isProcessing = ref(false)

// 图片处理相关
const beforeUpload = (file) => {
  const isImage = file.type.startsWith('image/')
  const isLt5M = file.size / 1024 / 1024 < 5

  if (!isImage) {
    ElMessage.error('只能上传图片文件!')
    return false
  }
  if (!isLt5M) {
    ElMessage.error('图片大小不能超过 5MB!')
    return false
  }

  return true
}

const customUpload = async (options) => {
  try {
    imageFile.value = options.file
    imageUrl.value = URL.createObjectURL(options.file)
  } catch (error) {
    console.error('上传失败:', error)
    ElMessage.error('图片上传失败，请重试')
  }
}

const analyzeImage = async () => {
  if (!imageFile.value) {
    ElMessage.error('请先上传图片')
    return
  }

  loading.value = true
  analysisProgress.value = 0
  startProgressAnimation()

  try {
    const formData = new FormData()
    formData.append('file', imageFile.value)
    formData.append('mode', analysisMode.value)

    console.log('发送分析请求...')
    const response = await imageRecognitionApi.analyzeImage(formData)
    console.log('收到分析响应:', response)
    
    if (response && (response.detected !== undefined || response.persons)) {
      analysisResult.value = response
      
      analysisHistoryStore.addAnalysis({
        id: Date.now(),
        timestamp: new Date().toISOString(),
        result: response,
        imageUrl: imageUrl.value
      })

      if (assistantRef.value) {
        assistantRef.value.notifyAnalysisComplete(response)
      }

      ElMessage.success('分析完成')
    } else {
      console.error('无效的响应数据:', response)
      ElMessage.error('分析结果格式不正确')
    }
  } catch (error) {
    console.error('分析失败:', error)
    ElMessage.error(error.message || '分析过程出错，请重试')
  } finally {
    loading.value = false
    stopProgressAnimation()
  }
}

// 缩放和平移方法
const zoomIn = () => {
  zoomLevel.value = Math.min(zoomLevel.value + 0.1, 3)
}

const zoomOut = () => {
  zoomLevel.value = Math.max(zoomLevel.value - 0.1, 0.5)
}

const resetZoom = () => {
  zoomLevel.value = 1
  panX.value = 0
  panY.value = 0
}

const startPan = (e) => {
  isPanning.value = true
  lastX.value = e.clientX
  lastY.value = e.clientY
}

const pan = (e) => {
  if (!isPanning.value) return
  
  const deltaX = e.clientX - lastX.value
  const deltaY = e.clientY - lastY.value
  
  panX.value += deltaX
  panY.value += deltaY
  
  lastX.value = e.clientX
  lastY.value = e.clientY
}

const endPan = () => {
  isPanning.value = false
}

// 人物标注相关
const getAnnotationStyle = (bbox) => {
  if (!bbox) return {}
  const [x, y, width, height] = bbox
  return {
    left: `${x}%`,
    top: `${y}%`,
    width: `${width}%`,
    height: `${height}%`
  }
}

const handlePersonClick = (index) => {
  analysisStore.setActivePerson(index)
}

// 查询处理
const handleQuery = async (query) => {
  const conditions = parseQuery(query)
  const matches = analysisStore.findMatchingPersons(conditions)
  
  if (matches.length > 0) {
    // 高亮所有匹配的人物
    matches.forEach((match, index) => {
      setTimeout(() => {
        analysisStore.setActivePerson(match.id)
      }, index * 1000) // 每隔1秒高亮一个人物
    })
    
    // 1秒后自动滚动到第一个匹配的人物的详细信息
    setTimeout(() => {
      const element = document.querySelector(`[name="${matches[0].id}"]`)
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }, 1000)
    
    // 生成描述
    const description = generateResultDescription(matches, conditions)
    ElMessage({
      message: description,
      type: 'success',
      duration: 5000,
      showClose: true
    })
  } else {
    ElMessage({
      message: '没有找到符合条件的人物',
      type: 'warning',
      duration: 3000
    })
  }
}

// 导出和分享功能
const exportResults = () => {
  if (!analysisResult.value) {
    ElMessage.warning('没有可导出的分析结果')
    return
  }
  
  // 准备导出数据
  const exportData = {
    timestamp: new Date().toISOString(),
    image_url: imageUrl.value,
    result_image_url: resultImageUrl.value,
    analysis_result: analysisResult.value
  }
  
  // 创建 Blob
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  
  // 创建下载链接
  const link = document.createElement('a')
  link.href = url
  link.download = `image-analysis-${new Date().getTime()}.json`
  document.body.appendChild(link)
  link.click()
  
  // 清理
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
  
  ElMessage.success('分析结果已导出')
}

const shareResults = async () => {
  if (!analysisResult.value) {
    ElMessage.warning('没有可分享的分析结果')
    return
  }
  
  try {
    // 创建分享数据
    const shareData = {
      title: '图片分析结果',
      text: `分析到 ${analysisResult.value.num_faces} 个人物`,
      url: window.location.href
    }
    
    // 尝试使用原生分享 API
    if (navigator.share) {
      await navigator.share(shareData)
      ElMessage.success('分享成功')
    } else {
      // 如果不支持原生分享，复制链接到剪贴板
      await navigator.clipboard.writeText(window.location.href)
      ElMessage.success('链接已复制到剪贴板')
    }
  } catch (error) {
    console.error('分享失败:', error)
    ElMessage.error('分享失败，请重试')
  }
}

// 颜色转换
const translateColor = (color) => {
  const colorMap = {
    'red': '红色',
    'blue': '蓝色',
    'green': '绿色',
    'yellow': '黄色',
    'black': '黑色',
    'white': '白色',
    'gray': '灰色',
    'purple': '紫色',
    'pink': '粉色',
    'brown': '棕色',
    'orange': '橙色',
    'navy': '深蓝色',
    'beige': '米色',
    'khaki': '卡其色',
    'unknown': '未知'
  }
  return colorMap[color] || color
}

// 添加性别翻译函数
const translateGender = (gender) => {
  const genderMap = {
    'male': '男',
    'female': '女',
    'unknown': '未知'
  }
  return genderMap[gender] || '未知'
}

// 修改颜色样式计算函数
const getColorStyle = (color) => {
  if (!color || color === 'unknown') return {}
  
  // 添加颜色映射
  const colorMap = {
    'red': '#ff0000',
    'blue': '#0000ff',
    'green': '#00ff00',
    'yellow': '#ffff00',
    'black': '#000000',
    'white': '#ffffff',
    'gray': '#808080',
    'purple': '#800080',
    'pink': '#ffc0cb',
    'brown': '#a52a2a',
    'orange': '#ffa500',
    'navy': '#000080',
    'beige': '#f5f5dc',
    'khaki': '#f0e68c'
  }

  const bgColor = colorMap[color] || color
  const textColor = isLightColor(bgColor) ? '#000000' : '#ffffff'

  return {
    backgroundColor: bgColor,
    color: textColor,
    padding: '2px 8px',
    borderRadius: '4px',
    display: 'inline-block'
  }
}

// 添加颜色亮度判断函数
const isLightColor = (color) => {
  // 移除 # 号
  const hex = color.replace('#', '')
  
  // 转换为 RGB
  const r = parseInt(hex.substr(0, 2), 16)
  const g = parseInt(hex.substr(2, 2), 16)
  const b = parseInt(hex.substr(4, 2), 16)
  
  // 计算亮度
  const brightness = ((r * 299) + (g * 587) + (b * 114)) / 1000
  return brightness > 128
}

// 生命周期钩子
onMounted(() => {
  window.addEventListener('mouseup', endPan)
})

onUnmounted(() => {
  window.removeEventListener('mouseup', endPan)
  if (progressInterval.value) {
    clearInterval(progressInterval.value)
  }
})

const startResize = (e) => {
  isResizing.value = true
  startX.value = e.clientX
  document.addEventListener('mousemove', resize)
  document.addEventListener('mouseup', endResize)
}

const resize = (e) => {
  if (!isResizing.value) return
  
  const dx = e.clientX - startX.value
  const containerWidth = document.querySelector('.main-container').offsetWidth
  const newWidth = Math.max(300, Math.min(containerWidth * 0.7, 
    parseInt(splitWidth.value) + dx))
  
  splitWidth.value = `${newWidth}px`
  startX.value = e.clientX
}

const endResize = () => {
  isResizing.value = false
  document.removeEventListener('mousemove', resize)
  document.removeEventListener('mouseup', endResize)
}

// 在组件卸载时清理事件监听
onUnmounted(() => {
  document.removeEventListener('mousemove', resize)
  document.removeEventListener('mouseup', endResize)
})

const hasResults = computed(() => {
  return analysisResult.value !== null
})

// 监听分析结果变化
watch(() => analysisResult.value, (newResult) => {
  if (newResult) {
    console.log('Analysis result updated:', newResult)
  }
})

// 添加人脸标注样式计算方法
const getFaceAnnotationStyle = (face) => {
  if (!face.face_bbox) return {}
  
  const [x1, y1, x2, y2] = face.face_bbox
  return {
    left: `${x1}px`,
    top: `${y1}px`,
    width: `${x2 - x1}px`,
    height: `${y2 - y1}px`
  }
}

// 添加图片URL处理函数
const getImageUrl = (url) => {
  if (!url) return ''
  // 确保URL以斜杠开头
  return url.startsWith('/') ? `http://localhost:8000${url}` : url
}

// 图片加载完成处理
const onImageLoad = () => {
  console.log('Result image loaded')
  // 重置缩放和平移
  resetZoom()
}

const startProgressAnimation = () => {
  // Implementation of startProgressAnimation
}

const stopProgressAnimation = () => {
  // Implementation of stopProgressAnimation
}

// 处理聊天提交
const handleChatSubmit = async () => {
  if (!chatInput.value.trim() || isProcessing.value) return
  
  const message = chatInput.value.trim()
  chatInput.value = ''
  isProcessing.value = true
  
  // 添加用户消息到聊天记录
  chatMessages.value.push({
    type: 'user',
    content: message
  })
  
  try {
    // 发送聊天请求，同时传递当前分析结果
    const response = await imageAnalysisChatApi.sendMessage(
      message,
      analysisResult.value
    )
    
    console.log('聊天响应:', response)
    
    if (response.data && response.data.content) {
      // 添加助手回复到聊天记录
      chatMessages.value.push({
        type: 'assistant',
        content: response.data.content
      })
      
      // 自动滚动到最新消息
      nextTick(() => {
        const chatContainer = document.querySelector('.chat-messages')
        if (chatContainer) {
          chatContainer.scrollTop = chatContainer.scrollHeight
        }
      })
    } else {
      ElMessage.error('获取回复失败')
      // 添加错误消息到聊天记录
      chatMessages.value.push({
        type: 'error',
        content: '获取回复失败，请重试'
      })
    }
  } catch (error) {
    console.error('聊天请求失败:', error)
    ElMessage.error(error.response?.data?.detail || '发送消息失败')
    
    // 添加错误消息到聊天记录
    chatMessages.value.push({
      type: 'error',
      content: '消息发送失败，请重试'
    })
  } finally {
    isProcessing.value = false
  }
}

// 添加聊天消息容器的引用
const chatMessagesRef = ref(null)

// 监听聊天消息变化，自动滚动到底部
watch(chatMessages, () => {
  nextTick(() => {
    const chatMessagesEl = chatMessagesRef.value
    if (chatMessagesEl) {
      chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight
    }
  })
}, { deep: true })
</script>

<style scoped>
.image-recognition {
  height: 100vh;
  overflow: hidden;
  background-color: var(--el-bg-color);
}

.main-container {
  height: 100%;
  padding: 20px;
}

.analysis-panel {
  height: 100%;
  overflow-y: auto;
  padding: 0;
  background-color: transparent;
}

.upload-card {
  margin-bottom: 20px;
}

.upload-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

.image-uploader {
  width: 400px;
  height: 300px;
  border: 1px dashed var(--el-border-color);
  border-radius: 8px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: var(--el-transition-duration);
}

.image-uploader:hover {
  border-color: var(--el-color-primary);
}

.upload-content {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.upload-icon {
  font-size: 28px;
  color: #8c939d;
}

.preview-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.toolbar {
  display: flex;
  gap: 10px;
  align-items: center;
}

.analyze-btn {
  width: 200px;
}

.result-card {
  margin-top: 20px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.result-content {
  display: flex;
  gap: 20px;
}

.result-image-container {
  flex: 1;
  max-width: 600px;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s ease;
}

.result-image {
  width: 100%;
  height: auto;
  border-radius: 4px;
  transition: transform 0.2s ease;
}

.annotations-layer {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.person-annotation {
  position: absolute;
  border: 2px solid var(--el-color-primary);
  border-radius: 4px;
  cursor: pointer;
  pointer-events: auto;
  transition: all 0.3s ease;
}

.person-annotation.active {
  border-color: var(--el-color-success);
  box-shadow: 0 0 10px rgba(0, 200, 0, 0.3);
  animation: highlight 1s ease-in-out infinite;
}

@keyframes highlight {
  0% {
    box-shadow: 0 0 10px rgba(0, 200, 0, 0.3);
  }
  50% {
    box-shadow: 0 0 20px rgba(0, 200, 0, 0.5);
  }
  100% {
    box-shadow: 0 0 10px rgba(0, 200, 0, 0.3);
  }
}

.annotation-label {
  position: absolute;
  top: -20px;
  left: -2px;
  background-color: var(--el-color-primary);
  color: white;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}

.result-details {
  flex: 1;
  min-width: 300px;
}

.person-info {
  padding: 10px;
}

:deep(.el-descriptions) {
  margin-bottom: 10px;
}

:deep(.el-tag) {
  margin-left: 8px;
}

/* 添加分隔线样式 */
.resizer {
  width: 4px;
  background-color: #e4e7ed;
  cursor: col-resize;
  transition: background-color 0.3s;
}

.resizer:hover {
  background-color: var(--el-color-primary);
}

/* 拖动时禁用文本选择 */
.image-recognition.resizing {
  user-select: none;
}

/* 修改颜色标签样式 */
.color-tag {
  margin-right: 8px;
  min-width: 60px;
  text-align: center;
  font-size: 14px;
}

/* 确保文字在浅色背景上显示清晰 */
.color-tag[data-color="white"],
.color-tag[data-color="yellow"],
.color-tag[data-color="beige"] {
  border: 1px solid #dcdfe6;
}

/* 确保文字在深色背景上显示清晰 */
.color-tag[data-color="black"],
.color-tag[data-color="navy"],
.color-tag[data-color="purple"] {
  border: 1px solid transparent;
}

.analysis-mode-section {
  margin: 20px 0;
  padding: 15px;
  background: var(--el-bg-color-overlay);
  border-radius: 8px;
}

.analysis-mode-section h3 {
  margin: 0 0 10px 0;
  font-size: 16px;
  color: var(--el-text-color-primary);
}

.mode-selector {
  width: 100%;
  display: flex;
  gap: 10px;
}

.progress-bar {
  margin-top: 20px;
}

.annotations {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.face-annotation {
  position: absolute;
  border: 2px solid #00ff9d;
  border-radius: 4px;
  pointer-events: all;
  cursor: pointer;
  transition: all 0.3s ease;
}

.face-id {
  position: absolute;
  top: -25px;
  left: -2px;
  background: #00ff9d;
  color: #000;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}

.face-info {
  display: none;
  position: absolute;
  top: 100%;
  left: 0;
  background: rgba(0, 0, 0, 0.8);
  padding: 8px;
  border-radius: 4px;
  font-size: 12px;
  color: #fff;
  white-space: nowrap;
}

.face-annotation:hover {
  border-color: #fff;
  z-index: 100;
}

.face-annotation:hover .face-info {
  display: block;
}

/* 修改聊天区域样式 */
.chat-section {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: transparent;
  position: relative;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.message {
  max-width: 80%;
  padding: 10px;
  border-radius: 8px;
  word-break: break-word;
}

.message.user {
  align-self: flex-end;
  background-color: var(--el-color-primary-light-9);
}

.message.assistant {
  align-self: flex-start;
  background-color: var(--el-bg-color-page);
  border: 1px solid var(--el-border-color-light);
}

.chat-input-container {
  padding: 20px;
  background: transparent;
  border-top: 1px solid var(--el-border-color-light);
}

/* 移除加载遮罩相关样式 */
.el-loading-mask {
  display: none !important;
}

/* 确保输入框在加载时不会被遮挡 */
.el-input.is-disabled .el-input__wrapper {
  background-color: var(--el-input-bg-color, var(--el-bg-color-overlay)) !important;
  box-shadow: none !important;
}

/* 移除任何可能的遮罩效果 */
.chat-section::before,
.chat-section::after {
  display: none;
}

.has-results {
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
}
</style> 