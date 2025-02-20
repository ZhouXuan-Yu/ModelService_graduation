<template>
  <div class="video-tracking">
    <h2 class="page-title">视频分析</h2>
    
    <div class="upload-section">
      <!-- 视频上传组件 -->
      <el-upload
        class="video-uploader"
        :action="null"
        :auto-upload="false"
        :on-change="handleVideoChange"
        :before-upload="beforeVideoUpload"
        accept="video/*"
        drag
      >
        <template #trigger>
          <el-button type="primary" :icon="Plus">选择视频</el-button>
        </template>
        <template #tip>
          <div class="el-upload__tip">
            支持 mp4、avi、mov 等常见视频格式，大小不超过 500MB
          </div>
        </template>
      </el-upload>

      <!-- 目标描述输入 -->
      <el-input
        v-model="description"
        type="textarea"
        :rows="3"
        placeholder="请描述视频分析需求，例如：分析视频中人物的年龄、性别和衣着特征"
        class="description-input"
      />
    </div>

    <!-- 视频预览和分析结果显示区域 -->
    <div v-if="videoUrl || resultVideoUrl" class="content-section">
      <div class="video-container">
        <video 
          ref="videoRef"
          controls
          :src="resultVideoUrl || videoUrl"
          @loadeddata="onVideoLoaded"
        ></video>
      </div>

      <!-- 分析控制按钮 -->
      <div class="control-panel">
        <el-button 
          type="primary" 
          @click="startAnalysis"
          :loading="isAnalyzing"
          :disabled="!videoFile"
          :icon="VideoPlay"
        >
          {{ isAnalyzing ? '分析中...' : '开始分析' }}
        </el-button>
      </div>

      <!-- 分析进度显示 -->
      <el-progress 
        v-if="isAnalyzing"
        :percentage="analysisProgress"
        :format="progressFormat"
        class="progress-bar"
        :stroke-width="20"
        status="success"
      />
    </div>

    <!-- 分析结果展示 -->
    <div v-if="analysisResult" class="analysis-result">
      <h3>分析结果</h3>
      <div class="result-stats">
        <el-descriptions :column="3" border>
          <el-descriptions-item label="总帧数">
            {{ analysisResult.total_frames }}
          </el-descriptions-item>
          <el-descriptions-item label="已处理帧数">
            {{ analysisResult.processed_frames }}
          </el-descriptions-item>
          <el-descriptions-item label="视频时长">
            {{ formatDuration(analysisResult.duration) }}
          </el-descriptions-item>
          <el-descriptions-item label="帧率">
            {{ analysisResult.fps }} FPS
          </el-descriptions-item>
          <el-descriptions-item label="分辨率">
            {{ analysisResult.resolution.width }} x {{ analysisResult.resolution.height }}
          </el-descriptions-item>
        </el-descriptions>
      </div>
      
      <div class="frame-results" v-if="analysisResult.frame_results">
        <h4>帧分析详情</h4>
        <el-collapse>
          <el-collapse-item 
            v-for="(frame, index) in analysisResult.frame_results" 
            :key="frame.frame_number"
            :title="'帧 ' + frame.frame_number + ' (' + formatTimestamp(frame.timestamp) + ')'"
          >
            <pre>{{ JSON.stringify(frame, null, 2) }}</pre>
          </el-collapse-item>
        </el-collapse>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { videoTrackingApi } from '@/api/videoTracking'
import { ElMessage } from 'element-plus'
import { Plus, VideoPlay } from '@element-plus/icons-vue'

// 状态变量
const videoRef = ref(null)
const videoUrl = ref('')
const resultVideoUrl = ref('')
const videoFile = ref(null)
const description = ref('')
const isAnalyzing = ref(false)
const analysisProgress = ref(0)
const analysisResult = ref(null)

// 处理视频文件选择
const handleVideoChange = (file) => {
  if (file) {
    videoFile.value = file.raw
    videoUrl.value = URL.createObjectURL(file.raw)
    // 重置相关状态
    resultVideoUrl.value = ''
    analysisResult.value = null
    analysisProgress.value = 0
    console.log('Video file selected:', file.raw.name)
  }
}

// 视频上传前的验证
const beforeVideoUpload = (file) => {
  const isVideo = file.type.startsWith('video/')
  const isLt500M = file.size / 1024 / 1024 < 500

  if (!isVideo) {
    ElMessage.error('请上传视频文件！')
    return false
  }
  if (!isLt500M) {
    ElMessage.error('视频大小不能超过 500MB！')
    return false
  }
  return true
}

// 视频加载完成的处理
const onVideoLoaded = () => {
  if (videoRef.value) {
    console.log('Video loaded:', {
      duration: videoRef.value.duration,
      width: videoRef.value.videoWidth,
      height: videoRef.value.videoHeight
    })
  }
}

// 开始分析
const startAnalysis = async () => {
  if (!videoFile.value) {
    ElMessage.warning('请先选择视频文件')
    return
  }

  if (!description.value.trim()) {
    ElMessage.warning('请输入分析需求描述')
    return
  }

  try {
    isAnalyzing.value = true
    analysisProgress.value = 0
    
    const formData = new FormData()
    formData.append('video', videoFile.value)
    formData.append('description', description.value)

    console.log('Starting video analysis...')

    // 模拟进度更新
    const progressInterval = setInterval(() => {
      if (analysisProgress.value < 90) {
        analysisProgress.value += 1
      }
    }, 1000)

    // 发送分析请求
    const result = await videoTrackingApi.analyzeVideo(formData)
    console.log('Analysis result:', result)
    
    clearInterval(progressInterval)
    analysisProgress.value = 100
    
    // 处理返回结果
    if (result.video_path) {
      resultVideoUrl.value = result.video_path
      console.log('Result video URL:', result.video_path)
    }
    analysisResult.value = result.analysis_result
    
    ElMessage.success('视频分析完成')
    
  } catch (error) {
    console.error('Analysis error:', error)
    ElMessage.error(error.message || '视频分析失败')
  } finally {
    isAnalyzing.value = false
  }
}

// 格式化进度显示
const progressFormat = (percentage) => {
  return `${percentage}% 已处理`
}

// 格式化时长
const formatDuration = (seconds) => {
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.floor(seconds % 60)
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
}

// 格式化时间戳
const formatTimestamp = (seconds) => {
  return formatDuration(seconds)
}

// 组件卸载时清理
onUnmounted(() => {
  if (videoUrl.value) {
    URL.revokeObjectURL(videoUrl.value)
  }
})
</script>

<style scoped>
.video-tracking {
  height: 100%;
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.page-title {
  margin: 0;
  padding: 0;
  color: var(--el-text-color-primary);
  font-size: 24px;
  font-weight: 500;
}

.upload-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 20px;
  background: var(--el-bg-color-overlay);
  border-radius: 8px;
}

.video-uploader {
  width: 100%;
}

.description-input {
  width: 100%;
}

.content-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.video-container {
  width: 100%;
  background: var(--el-bg-color-overlay);
  border-radius: 8px;
  overflow: hidden;
}

.video-container video {
  width: 100%;
  max-height: 600px;
  object-fit: contain;
}

.control-panel {
  display: flex;
  justify-content: center;
  gap: 20px;
  padding: 20px 0;
}

.progress-bar {
  padding: 0 20px;
}

.analysis-result {
  padding: 20px;
  background: var(--el-bg-color-overlay);
  border-radius: 8px;
  margin-top: 20px;
}

.analysis-result h3 {
  margin-top: 0;
  margin-bottom: 20px;
  color: var(--el-text-color-primary);
}

.result-stats {
  margin-bottom: 20px;
}

.frame-results {
  margin-top: 20px;
}

.frame-results h4 {
  margin-bottom: 15px;
  color: var(--el-text-color-primary);
}

.frame-results pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  color: var(--el-text-color-primary);
  background: var(--el-bg-color);
  padding: 10px;
  border-radius: 4px;
  font-size: 14px;
  line-height: 1.5;
}

:deep(.el-upload-dragger) {
  background: var(--el-bg-color);
  border-color: var(--el-border-color-light);
}

:deep(.el-upload-dragger:hover) {
  border-color: var(--el-color-primary);
}

:deep(.el-descriptions) {
  --el-descriptions-item-bordered-label-background: var(--el-bg-color);
}

:deep(.el-collapse-item__header) {
  background: var(--el-bg-color);
  border-color: var(--el-border-color-light);
}

:deep(.el-collapse-item__content) {
  background: var(--el-bg-color-overlay);
}
</style> 