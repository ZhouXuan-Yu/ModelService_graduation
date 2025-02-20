// components/layout/NotificationSystem.vue
<template>
  <div class="notification-system">
    <TransitionGroup name="notification">
      <div
        v-for="notification in notifications"
        :key="notification.id"
        class="notification"
        :class="notification.type"
      >
        <el-icon class="notification-icon">
          <component :is="getIcon(notification.type)" />
        </el-icon>
        <div class="notification-content">
          <div class="notification-title">{{ notification.title }}</div>
          <div class="notification-message">{{ notification.message }}</div>
        </div>
        <el-button
          class="close-button"
          circle
          text
          @click="removeNotification(notification.id)"
        >
          <el-icon><Close /></el-icon>
        </el-button>
      </div>
    </TransitionGroup>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { 
  SuccessFilled,
  WarningFilled,
  CircleCloseFilled,
  InfoFilled,
  Close
} from '@element-plus/icons-vue'

const notifications = ref([])

const getIcon = (type) => {
  switch (type) {
    case 'success':
      return SuccessFilled
    case 'warning':
      return WarningFilled
    case 'error':
      return CircleCloseFilled
    default:
      return InfoFilled
  }
}

const addNotification = (notification) => {
  const id = Date.now()
  notifications.value.push({
    id,
    ...notification
  })

  setTimeout(() => {
    removeNotification(id)
  }, notification.duration || 3000)
}

const removeNotification = (id) => {
  const index = notifications.value.findIndex(n => n.id === id)
  if (index !== -1) {
    notifications.value.splice(index, 1)
  }
}

defineExpose({
  addNotification
})
</script>

<style scoped>
.notification-system {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 10px;
  pointer-events: none;
}

.notification {
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 12px;
  min-width: 300px;
  display: flex;
  align-items: flex-start;
  gap: 12px;
  pointer-events: auto;
}

.notification.success {
  border-left: 4px solid #67c23a;
}

.notification.warning {
  border-left: 4px solid #e6a23c;
}

.notification.error {
  border-left: 4px solid #f56c6c;
}

.notification.info {
  border-left: 4px solid #909399;
}

.notification-icon {
  font-size: 20px;
  margin-top: 2px;
}

.notification-content {
  flex: 1;
}

.notification-title {
  font-weight: bold;
  margin-bottom: 4px;
  color: #ffffff;
}

.notification-message {
  color: rgba(255, 255, 255, 0.8);
  font-size: 14px;
}

.close-button {
  color: rgba(255, 255, 255, 0.6);
}

.close-button:hover {
  color: #ffffff;
}

.notification-enter-active,
.notification-leave-active {
  transition: all 0.3s ease;
}

.notification-enter-from {
  opacity: 0;
  transform: translateX(30px);
}

.notification-leave-to {
  opacity: 0;
  transform: translateX(30px);
}
</style>