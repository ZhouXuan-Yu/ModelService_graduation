<template>
  <div class="image-recognition-view">
    <el-container>
      <el-main class="main-content" :style="{ width: `calc(100% - ${rightPanelWidth}px - 4px)` }">
        <ImageRecognition @analysis-complete="handleAnalysisComplete" />
      </el-main>
      
      <div class="splitter"
           @mousedown="startResize">
        <div class="splitter-handle"></div>
      </div>
      
      <el-aside :width="`${rightPanelWidth}px`" class="assistant-panel">
        <ImageAnalysisAssistant 
          ref="assistantRef"
          :analysis-result="analysisResult"
          @query="handleAssistantQuery"
        />
      </el-aside>
    </el-container>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import ImageRecognition from './ImageRecognition.vue'
import ImageAnalysisAssistant from '@/components/ImageAnalysisAssistant.vue'
import { useAnalysisStore } from '@/stores/analysis'
import { ElMessage } from 'element-plus'

const analysisStore = useAnalysisStore()
const assistantRef = ref(null)
const analysisResult = ref(null)

// 分隔线拖动相关
const rightPanelWidth = ref(400)
const isResizing = ref(false)
const startX = ref(0)
const startWidth = ref(0)

const analysisMode = ref('normal')

const startResize = (e) => {
  isResizing.value = true
  startX.value = e.clientX
  startWidth.value = rightPanelWidth.value
  document.body.style.cursor = 'col-resize'
  document.addEventListener('mousemove', handleResize)
  document.addEventListener('mouseup', endResize)
}

const handleResize = (e) => {
  if (!isResizing.value) return
  
  const dx = startX.value - e.clientX
  const newWidth = Math.min(Math.max(300, startWidth.value + dx), window.innerWidth * 0.7)
  rightPanelWidth.value = newWidth
}

const endResize = () => {
  isResizing.value = false
  document.body.style.cursor = ''
  document.removeEventListener('mousemove', handleResize)
  document.removeEventListener('mouseup', endResize)
}

// 处理分析完成事件
const handleAnalysisComplete = (result) => {
  analysisResult.value = result
  // 通知助手分析已完成
  if (assistantRef.value) {
    assistantRef.value.notifyAnalysisComplete(result)
  }
}

// 处理助手的查询
const handleAssistantQuery = (query) => {
  // 高亮匹配的人物
  if (query.matches && query.matches.length > 0) {
    query.matches.forEach((match, index) => {
      setTimeout(() => {
        analysisStore.setActivePerson(match.id)
      }, index * 1000)
    })
  }
}

// 清理事件监听
onUnmounted(() => {
  document.removeEventListener('mousemove', handleResize)
  document.removeEventListener('mouseup', endResize)
})

const analyzeImage = async () => {
  if (!imageFile.value) {
    ElMessage.warning('请先上传图片')
    return
  }

  loading.value = true
  try {
    const formData = new FormData()
    formData.append('image', imageFile.value)
    formData.append('description', 'basic analysis')
    formData.append('mode', analysisMode.value)

    const response = await imageRecognitionApi.analyzeImage(formData)
    
    // 更新状态
    analysisResult.value = response
    analysisStore.setAnalysisResult(response)
    resultImageUrl.value = response.result_image_url
    
    // 通知助手分析完成
    if (assistantRef.value) {
      assistantRef.value.notifyAnalysisComplete(response, analysisMode.value)
    }
    
    ElMessage.success(`${analysisMode.value === 'enhanced' ? '加强' : '普通'}分析完成`)
  } catch (error) {
    console.error('分析失败:', error)
    ElMessage.error('分析失败，请重试')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.image-recognition-view {
  height: 100vh;
  background-color: var(--el-bg-color);
  overflow: hidden;
}

.el-container {
  height: 100%;
  position: relative;
  display: flex;
}

.main-content {
  padding: 20px;
  height: 100%;
  overflow-y: auto;
  flex: none;
  background-color: var(--el-bg-color);
}

.splitter {
  width: 4px;
  background-color: #1a1a1a;
  cursor: col-resize;
  display: flex;
  justify-content: center;
  align-items: center;
  flex: none;
}

.splitter-handle {
  width: 4px;
  height: 30px;
  background-color: rgba(255, 255, 255, 0.2);
  border-radius: 2px;
  transition: background-color 0.3s;
}

.splitter:hover .splitter-handle {
  background-color: rgba(255, 255, 255, 0.4);
}

.assistant-panel {
  height: 100%;
  overflow: hidden;
  flex: none;
  background-color: #141414;
  min-width: 300px;
}

/* 拖动时禁用文本选择 */
.image-recognition-view.resizing {
  user-select: none;
}

:deep(.el-main) {
  padding: 0;
}

:deep(.el-aside) {
  overflow: hidden;
}
</style>