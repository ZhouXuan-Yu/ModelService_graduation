// components/layout/AppHeader.vue
<template>
  <el-header class="header">
    <div class="header-left">
      <h2 class="header-title">慧眼智程 AI </h2>
      <div class="breadcrumb">
        <el-breadcrumb separator="/">
          <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>
          <el-breadcrumb-item>{{ currentRoute }}</el-breadcrumb-item>
        </el-breadcrumb>
      </div>
    </div>

    <div class="header-actions">
      <div class="search-bar">
        <el-input
          v-model="searchQuery"
          placeholder="搜索..."
          prefix-icon="Search"
        >
          <template #suffix>
            <el-tooltip content="快捷键: Ctrl + K">
              <kbd>⌘K</kbd>
            </el-tooltip>
          </template>
        </el-input>
      </div>

      <div class="action-buttons">
        <el-button 
          type="success" 
          class="action-button"
          @mouseenter="$emit('buttonHover', '#00ff9d')"
          @click="showNewTaskDialog = true"
        >
          <el-icon><Plus /></el-icon>新建任务
        </el-button>
        
        <el-button 
          type="info" 
          class="action-button secondary"
          @mouseenter="$emit('buttonHover', '#00d6ff')"
        >
          <el-icon><Setting /></el-icon>系统设置
        </el-button>

        <el-badge :value="3" class="notification-badge">
          <el-button 
            type="info" 
            class="action-button secondary"
            @click="showNotifications = true"
          >
            <el-icon><Bell /></el-icon>
          </el-button>
        </el-badge>
      </div>
    </div>

    <!-- New Task Dialog -->
    <el-dialog
      v-model="showNewTaskDialog"
      title="新建任务"
      width="50%"
      class="custom-dialog"
    >
      <el-form :model="newTaskForm" label-width="100px">
        <el-form-item label="任务类型">
          <el-select v-model="newTaskForm.type" placeholder="请选择任务类型">
            <el-option label="路径规划" value="route" />
            <el-option label="图像识别" value="image" />
            <el-option label="视频追踪" value="video" />
          </el-select>
        </el-form-item>
        <el-form-item label="任务名称">
          <el-input v-model="newTaskForm.name" placeholder="请输入任务名称" />
        </el-form-item>
        <el-form-item label="任务描述">
          <el-input 
            v-model="newTaskForm.description" 
            type="textarea" 
            :rows="3"
            placeholder="请输入任务描述"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showNewTaskDialog = false">取消</el-button>
        <el-button type="primary" @click="createNewTask">确认</el-button>
      </template>
    </el-dialog>

    <!-- Notifications Drawer -->
    <el-drawer
      v-model="showNotifications"
      title="通知中心"
      direction="rtl"
      size="400px"
    >
      <div class="notifications-container">
        <div 
          v-for="notification in notifications" 
          :key="notification.id"
          class="notification-item"
        >
          <div class="notification-icon" :class="notification.type">
            <el-icon><component :is="notification.icon" /></el-icon>
          </div>
          <div class="notification-content">
            <h4>{{ notification.title }}</h4>
            <p>{{ notification.message }}</p>
            <span class="notification-time">{{ notification.time }}</span>
          </div>
        </div>
      </div>
    </el-drawer>
  </el-header>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import { 
  Plus,
  Setting,
  Bell,
  Search,
  InfoFilled,
  WarningFilled,
  CircleCheckFilled
} from '@element-plus/icons-vue'

const route = useRoute()
const searchQuery = ref('')
const showNewTaskDialog = ref(false)
const showNotifications = ref(false)

const newTaskForm = ref({
  type: '',
  name: '',
  description: ''
})

const notifications = [
  {
    id: 1,
    type: 'success',
    icon: CircleCheckFilled,
    title: '任务完成',
    message: '路径规划任务已成功完成',
    time: '10分钟前'
  },
  {
    id: 2,
    type: 'warning',
    icon: WarningFilled,
    title: '系统提醒',
    message: '系统资源使用率较高',
    time: '30分钟前'
  },
  {
    id: 3,
    type: 'info',
    icon: InfoFilled,
    title: '新功能上线',
    message: '新版本图像识别功能已上线',
    time: '2小时前'
  }
]

const currentRoute = computed(() => {
  const routeMap = {
    '/route-planning': '智能路径规划',
    '/image-recognition': '高级图像识别',
    '/video-tracking': '实时视频追踪'
  }
  return routeMap[route.path] || '首页'
})

const emit = defineEmits(['buttonHover', 'showNotification'])

const createNewTask = () => {
  showNewTaskDialog.value = false
  emit('showNotification', {
    type: 'success',
    title: '创建成功',
    message: '新任务已创建',
    duration: 3000
  })
}
</script>

<style scoped>
.header {
  background: rgba(13, 18, 23, 0.7);
  backdrop-filter: blur(20px);
  height: 80px !important;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.header-left {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.header-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: #ffffff;
  margin: 0;
  text-shadow: 0 0 20px rgba(0, 255, 157, 0.5);
}

.breadcrumb {
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.9rem;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 24px;
}

.search-bar {
  width: 300px;
}

.search-bar :deep(.el-input__wrapper) {
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: none;
}

.search-bar :deep(.el-input__inner) {
  color: #ffffff;
}

.search-bar :deep(.el-input__inner::placeholder) {
  color: rgba(255, 255, 255, 0.5);
}

kbd {
  background: rgba(255, 255, 255, 0.1);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.8);
}

.action-buttons {
  display: flex;
  gap: 16px;
}

.action-button {
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 8px 20px;
  height: 40px;
  display: flex;
  align-items: center;
  gap: 8px;
  border-radius: 20px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.action-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 0 20px rgba(0, 255, 157, 0.3);
  border-color: #00ff9d;
}

.action-button.secondary:hover {
  box-shadow: 0 0 20px rgba(0, 214, 255, 0.3);
  border-color: #00d6ff;
}

.notification-badge :deep(.el-badge__content) {
  background-color: #ff0066;
}

.notifications-container {
  padding: 16px;
  height: 100%;
  overflow-y: auto;
  background: #1a1a1a;
}

.notification-item {
  position: relative;
  z-index: 20001 !important;
  display: flex;
  gap: 16px;
  padding: 16px;
  margin-bottom: 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.05);
  transition: all 0.3s ease;
}

.notification-item:hover {
  background: rgba(255, 255, 255, 0.08);
  transform: translateX(-4px);
}

.notification-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.notification-icon.success {
  background: rgba(0, 255, 157, 0.1);
  color: #00ff9d;
}

.notification-icon.warning {
  background: rgba(255, 214, 0, 0.1);
  color: #ffd600;
}

.notification-icon.info {
  background: rgba(0, 214, 255, 0.1);
  color: #00d6ff;
}

.notification-content {
  flex: 1;
}

.notification-content h4 {
  margin: 0 0 8px 0;
  color: #ffffff;
  font-size: 16px;
  font-weight: 500;
}

.notification-content p {
  margin: 0;
  color: rgba(255, 255, 255, 0.8);
  font-size: 14px;
  line-height: 1.5;
}

.notification-time {
  display: block;
  margin-top: 8px;
  color: rgba(255, 255, 255, 0.5);
  font-size: 12px;
}

.custom-dialog {
  background: rgba(20, 25, 30, 0.95);
  backdrop-filter: blur(20px);
}

.custom-dialog :deep(.el-dialog__header) {
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.custom-dialog :deep(.el-dialog__footer) {
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

/* 修改抽屉相关样式，设置超高优先级 */
:deep(.el-drawer__wrapper) {
  position: fixed !important;
  /* 设置一个非常高的z-index，确保高于其他所有模块 */
  z-index: 20000 !important;
}

:deep(.el-drawer) {
  position: fixed !important;
  z-index: 20000 !important;
  background: #1a1a1a !important;
  border-left: 1px solid rgba(255, 255, 255, 0.1);
  height: 100vh !important;
}

/* 遮罩层也需要更高的z-index */
:deep(.el-overlay) {
  z-index: 19999 !important;
}

/* 抽屉内容容器 */
.notifications-container {
  position: relative;
  z-index: 20001 !important; /* 确保内容在最上层 */
  padding: 16px;
  height: 100%;
  background: #1a1a1a;
}

/* 通知项样式 */
.notification-item {
  position: relative;
  z-index: 20001 !important;
  display: flex;
  gap: 16px;
  padding: 16px;
  margin-bottom: 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.05);
  transition: all 0.3s ease;
}

/* 抽屉头部样式 */
:deep(.el-drawer__header) {
  position: relative;
  z-index: 20001 !important;
  margin-bottom: 0;
  padding: 16px 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  background: #1a1a1a;
}

/* 移除重复的样式定义，保留一个位置 */
.notifications-container {
  padding: 16px;
  height: 100%;
  overflow-y: auto;
  background: #1a1a1a;
}

/* 确保关闭按钮可见 */
:deep(.el-drawer__close-btn) {
  position: relative;
  z-index: 20002 !important;
  color: rgba(255, 255, 255, 0.7);
}

/* 移除可能冲突的较低 z-index 设置 */
:deep(.v-modal) {
  display: none !important; /* 如果遮罩层导致问题，可以直接移除 */
}
</style>