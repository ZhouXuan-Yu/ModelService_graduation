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
                 ref="containerRef"
                 :style="{ transform: `scale(${zoomLevel})` }"
                 @mousedown="startPan"
                 @mousemove="pan"
                 @mouseup="endPan"
                 @mouseleave="endPan">
              <img 
                v-if="imageUrl"
                ref="imageRef"
                :src="imageUrl"
                alt="分析结果"
                class="result-image"
                :style="{
                  transform: `translate(${panX}px, ${panY}px)`,
                  width: imageDisplaySize.width ? `${imageDisplaySize.width}px` : 'auto',
                  height: imageDisplaySize.height ? `${imageDisplaySize.height}px` : 'auto',
                  objectFit: 'contain'
                }"
                @load="onImageLoad"
              />
              <!-- 人物标注层 -->
              <div class="annotations-layer"
                   :style="{
                     transform: `translate(${panX}px, ${panY}px)`,
                     width: imageDisplaySize.width ? `${imageDisplaySize.width}px` : '100%',
                     height: imageDisplaySize.height ? `${imageDisplaySize.height}px` : '100%'
                   }">
                <div v-for="(person, index) in analysisResult?.persons" 
                     :key="index"
                     class="person-annotation"
                     :class="{ 
                       active: activePersonId === index,
                       'has-data': person.gender !== 'unknown' || person.age > 0 
                     }"
                     :style="calculateBoxStyle(person.bbox)"
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
  if (percentage < 30) {
    return `${percentage}% 正在初始化分析...`
  } else if (percentage < 60) {
    return `${percentage}% 正在处理图像...`
  } else if (percentage < 90) {
    return `${percentage}% 正在生成结果...`
  } else {
    return `${percentage}% 即将完成...`
  }
}

// 添加 assistantRef
const assistantRef = ref(null)

// 添加聊天相关的状态
const chatInput = ref('')
const chatMessages = ref([])
const isProcessing = ref(false)

// 添加图片尺寸状态
const imageSize = ref({ width: 0, height: 0 })
const containerSize = ref({ width: 0, height: 0 })

// 添加图片缩放比例
const imageScale = ref(1)

// 添加图片显示尺寸状态
const imageDisplaySize = ref({ width: 0, height: 0 })

// 在 script setup 中添加新的 ref
const imageRef = ref(null)
const containerRef = ref(null)

// 添加防抖函数
const debounce = (fn, delay) => {
  let timer = null
  return function (...args) {
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      fn.apply(this, args)
    }, delay)
  }
}

// 使用防抖优化更新
const debouncedUpdateBoxes = debounce(() => {
  if (analysisResult.value) {
    analysisResult.value = { ...analysisResult.value }
  }
}, 16) // 约60fps的更新频率

// 修改 ResizeObserver 回调
const resizeObserver = new ResizeObserver(() => {
  if (containerRef.value && imageRef.value) {
    nextTick(() => {
      calculateImageDisplaySize()
      debouncedUpdateBoxes()
    })
  }
})

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
  // 只有在没有标注框被点击的情况下才允许平移
  if (e.target.classList.contains('person-annotation')) return
  
  isPanning.value = true
  lastX.value = e.clientX
  lastY.value = e.clientY
}

const pan = (e) => {
  if (!isPanning.value) return
  
  const deltaX = e.clientX - lastX.value
  const deltaY = e.clientY - lastY.value
  
  // 限制平移范围
  const maxPanX = imageDisplaySize.value.width * (zoomLevel.value - 1) / 2
  const maxPanY = imageDisplaySize.value.height * (zoomLevel.value - 1) / 2
  
  panX.value = Math.max(-maxPanX, Math.min(maxPanX, panX.value + deltaX))
  panY.value = Math.max(-maxPanY, Math.min(maxPanY, panY.value + deltaY))
  
  lastX.value = e.clientX
  lastY.value = e.clientY
}

const endPan = () => {
  isPanning.value = false
}

// 人物标注相关
const calculateBoxStyle = (bbox) => {
  if (!bbox || !imageRef.value || !containerRef.value) {
    console.warn('缺少计算标记框所需的参数')
    return {}
  }

  try {
    const [x1, y1, x2, y2] = bbox
    const img = imageRef.value
    const container = containerRef.value
    
    // 获取图片的原始尺寸
    const imgNaturalWidth = img.naturalWidth
    const imgNaturalHeight = img.naturalHeight
    
    // 获取容器的实际尺寸
    const containerRect = container.getBoundingClientRect()
    const containerWidth = containerRect.width
    const containerHeight = containerRect.height
    
    // 计算图片在容器中的实际显示尺寸
    const imageRatio = imgNaturalWidth / imgNaturalHeight
    const containerRatio = containerWidth / containerHeight
    
    let displayWidth, displayHeight, offsetX = 0, offsetY = 0
    
    if (imageRatio > containerRatio) {
      // 图片较宽，以容器宽度为准
      displayWidth = containerWidth
      displayHeight = containerWidth / imageRatio
      offsetY = (containerHeight - displayHeight) / 2
    } else {
      // 图片较高，以容器高度为准
      displayHeight = containerHeight
      displayWidth = containerHeight * imageRatio
      offsetX = (containerWidth - displayWidth) / 2
    }
    
    // 计算缩放比例
    const scaleX = displayWidth / imgNaturalWidth
    const scaleY = displayHeight / imgNaturalHeight
    
    // 计算边界框在实际显示中的位置和尺寸
    const boxLeft = x1 * scaleX + offsetX
    const boxTop = y1 * scaleY + offsetY
    const boxWidth = (x2 - x1) * scaleX
    const boxHeight = (y2 - y1) * scaleY
    
    // 验证计算结果
    if ([boxLeft, boxTop, boxWidth, boxHeight].some(val => isNaN(val))) {
      console.error('标记框计算结果无效:', { boxLeft, boxTop, boxWidth, boxHeight })
      return {}
    }
    
    // 返回标记框样式
    return {
      position: 'absolute',
      left: `${boxLeft}px`,
      top: `${boxTop}px`,
      width: `${boxWidth}px`,
      height: `${boxHeight}px`,
      transform: 'none',
      backgroundColor: 'rgba(64, 158, 255, 0.1)',
      backdropFilter: 'blur(2px)',
      border: '2px solid var(--el-color-primary)',
      borderRadius: '8px',
      zIndex: '2',
      pointerEvents: 'auto',
      cursor: 'pointer',
      transition: 'all 0.3s ease'
    }
  } catch (error) {
    console.error('计算标记框样式出错:', error)
    return {}
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

// 图片加载完成后的处理
const onImageLoad = async () => {
  try {
    await nextTick()
    if (!imageRef.value || !containerRef.value) {
      console.warn('图片或容器引用未找到')
      return
    }
    
    const img = imageRef.value
    const container = containerRef.value
    
    // 禁止图片拖动
    img.draggable = false
    
    // 获取容器尺寸
    const containerRect = container.getBoundingClientRect()
    
    // 保存图片原始尺寸
    imageSize.value = {
      width: img.naturalWidth,
      height: img.naturalHeight
    }
    
    // 保存容器尺寸
    containerSize.value = {
      width: containerRect.width,
      height: containerRect.height
    }
    
    // 计算图片显示尺寸
    const imageRatio = imageSize.value.width / imageSize.value.height
    const containerRatio = containerSize.value.width / containerSize.value.height
    
    let displayWidth, displayHeight
    
    if (imageRatio > containerRatio) {
      displayWidth = containerSize.value.width
      displayHeight = displayWidth / imageRatio
    } else {
      displayHeight = containerSize.value.height
      displayWidth = displayHeight * imageRatio
    }
    
    // 更新图片显示尺寸
    imageDisplaySize.value = {
      width: displayWidth,
      height: displayHeight
    }
    
    // 计算缩放比例
    imageScale.value = displayWidth / imageSize.value.width
    
    // 强制更新标记框位置
    if (analysisResult.value) {
      nextTick(() => {
        analysisResult.value = { ...analysisResult.value }
      })
    }
    
    console.log('图片加载完成:', {
      naturalSize: imageSize.value,
      containerSize: containerSize.value,
      displaySize: imageDisplaySize.value,
      scale: imageScale.value
    })
  } catch (error) {
    console.error('图片加载处理出错:', error)
    ElMessage.error('图片加载失败，请重试')
  }
}

// 监听窗口大小变化
const handleResize = debounce(() => {
  if (imageRef.value && containerRef.value) {
    onImageLoad()
  }
}, 200)

// 组件挂载时添加监听器
onMounted(() => {
  window.addEventListener('resize', handleResize)
})

// 组件卸载时移除监听器
onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
})

// 添加图片显示尺寸计算函数
const calculateImageDisplaySize = () => {
  if (!imageSize.value.width || !containerSize.value.width) return
  
  const containerRatio = containerSize.value.width / containerSize.value.height
  const imageRatio = imageSize.value.width / imageSize.value.height
  
  let displayWidth, displayHeight
  
  if (imageRatio > containerRatio) {
    // 图片较宽，以容器宽度为准
    displayWidth = containerSize.value.width
    displayHeight = displayWidth / imageRatio
  } else {
    // 图片较高，以容器高度为准
    displayHeight = containerSize.value.height
    displayHeight = Math.min(displayHeight, containerSize.value.height * 0.9) // 留出一些边距
    displayWidth = displayHeight * imageRatio
  }
  
  imageDisplaySize.value = {
    width: displayWidth,
    height: displayHeight
  }
  
  // 计算缩放比例
  imageScale.value = displayWidth / imageSize.value.width
  
  console.log('图片显示尺寸计算结果:', {
    containerSize: containerSize.value,
    imageSize: imageSize.value,
    displaySize: imageDisplaySize.value,
    scale: imageScale.value
  })
}

// 监听缩放级别变化
watch(zoomLevel, () => {
  nextTick(() => {
    // 强制更新标记框位置
    analysisResult.value = { ...analysisResult.value }
  })
})

const startProgressAnimation = () => {
  analysisProgress.value = 0
  let baseProgress = 0
  let speed = 1

  // 清除可能存在的旧定时器
  if (progressInterval.value) {
    clearInterval(progressInterval.value)
  }

  progressInterval.value = setInterval(() => {
    if (baseProgress < 90) { // 只增长到90%，留出实际完成的空间
      // 动态调整速度，进度越大，速度越慢
      if (baseProgress < 30) {
        speed = 2
      } else if (baseProgress < 60) {
        speed = 1
      } else {
        speed = 0.5
      }

      baseProgress += speed
      analysisProgress.value = Math.min(90, baseProgress)
    }
  }, 100)
}

const stopProgressAnimation = () => {
  if (progressInterval.value) {
    clearInterval(progressInterval.value)
    progressInterval.value = null
  }
  
  // 平滑过渡到100%
  analysisProgress.value = 90
  setTimeout(() => {
    analysisProgress.value = 95
  }, 100)
  setTimeout(() => {
    analysisProgress.value = 100
  }, 200)
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
  width: 100%;
  height: 100%;
  gap: 20px;
  position: relative;
  overflow: hidden;
  padding: 0 20px;
}

.result-image-container {
  flex: 0 0 65%;
  position: relative;
  height: 600px;
  overflow: hidden;
  display: flex;
  justify-content: flex-start;
  align-items: center;
  background-color: rgba(0, 0, 0, 0.1);
  user-select: none;
  margin-right: 20px;
}

.result-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  margin-left: 20px;
}

.annotations-layer {
  position: absolute;
  top: 0;
  left: 20px;
  width: calc(100% - 20px);
  height: 100%;
  pointer-events: none;
  z-index: 10;
}

.person-annotation {
  position: absolute;
  border: 2px solid var(--el-color-primary);
  border-radius: 8px;
  pointer-events: auto;
  cursor: pointer;
  transition: all 0.3s ease;
  z-index: 11;
  background-color: rgba(64, 158, 255, 0.1);
  backdrop-filter: blur(2px);
}

.annotation-label {
  position: absolute;
  top: -30px;
  left: 50%;
  transform: translateX(-50%);
  background-color: var(--el-color-primary);
  color: white;
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 14px;
  font-weight: bold;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  z-index: 12;
  white-space: nowrap;
  pointer-events: none;
}

.result-details {
  flex: 0 0 35%;
  min-width: 400px;
  max-width: 500px;
  overflow-y: auto;
  padding: 20px;
  background: var(--el-bg-color-overlay);
  border-left: 1px solid var(--el-border-color-light);
  z-index: 20;
  position: relative;
}

.person-info {
  padding: 15px;
  margin-bottom: 15px;
}

:deep(.el-descriptions) {
  margin-bottom: 15px;
}

:deep(.el-descriptions__cell) {
  padding: 12px 15px;
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

/* 修改描述列表的样式 */
:deep(.el-descriptions__cell.el-descriptions__label) {
  background-color: #141414 !important; /* 使用深色背景 */
  color: rgba(255, 255, 255, 0.85) !important; /* 使用浅色文字 */
}

:deep(.el-descriptions__cell.el-descriptions__content) {
  background-color: #141414 !important;
  color: rgba(255, 255, 255, 0.85) !important;
}

/* 确保边框样式正确 */
:deep(.el-descriptions) {
  --el-descriptions-item-bordered-label-background: #141414;
}

/* 标签和内容的边框颜色 */
:deep(.el-descriptions--bordered .el-descriptions__cell) {
  border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* 确保颜色标签在深色背景下可见 */
.color-tag {
  padding: 2px 8px;
  border-radius: 4px;
  margin-right: 8px;
  display: inline-block;
}
</style> 