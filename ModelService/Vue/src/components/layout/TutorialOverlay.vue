// components/layout/TutorialOverlay.vue
<template>
  <div class="tutorial-overlay">
    <div class="tutorial-content">
      <h2>欢迎使用 AI Vision Pro</h2>
      <div class="tutorial-steps">
        <div class="step">
          <el-icon><Location /></el-icon>
          <div class="step-content">
            <h3>智能路径规划</h3>
            <p>使用自然语言描述您的路线需求，AI 将为您规划最优路线。</p>
          </div>
        </div>
        
        <div class="step">
          <el-icon><Picture /></el-icon>
          <div class="step-content">
            <h3>高级图像识别</h3>
            <p>上传图片或使用摄像头，AI 将识别并分析图像中的内容。</p>
          </div>
        </div>
        
        <div class="step">
          <el-icon><VideoCamera /></el-icon>
          <div class="step-content">
            <h3>实时视频追踪</h3>
            <p>通过视频流实时追踪和分析目标对象的运动轨迹。</p>
          </div>
        </div>
      </div>
      
      <div class="tutorial-actions">
        <el-button type="primary" @click="$emit('close')">开始使用</el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { Location, Picture, VideoCamera } from '@element-plus/icons-vue'

defineEmits(['close'])
</script>

<style scoped>
.tutorial-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.tutorial-content {
  background: rgba(30, 35, 40, 0.95);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 32px;
  max-width: 600px;
  width: 90%;
}

.tutorial-content h2 {
  text-align: center;
  color: #ffffff;
  margin: 0 0 32px 0;
  font-size: 24px;
}

.tutorial-steps {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.step {
  display: flex;
  align-items: flex-start;
  gap: 16px;
}

.step .el-icon {
  font-size: 24px;
  color: #00ff9d;
  margin-top: 4px;
}

.step-content h3 {
  color: #ffffff;
  margin: 0 0 8px 0;
  font-size: 18px;
}

.step-content p {
  color: rgba(255, 255, 255, 0.8);
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
}

.tutorial-actions {
  margin-top: 32px;
  text-align: center;
}
</style>